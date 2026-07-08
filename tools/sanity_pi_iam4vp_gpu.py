"""Single-GPU deep sanity diagnostics for PI-IAM4VP physics arms on real ERA5 data.

Runs every v4 PI-IAM4VP arm (plain AI, legacy latent, residual fixedeq /
legacy_hybrid / no_physics / massconsistent / diabatic) through a short
train-like rollout on a real memmap batch and verifies the invariants that
production training relies on:

* forward/backward finiteness of loss, aux loss and every diagnostic;
* gradient health: no nonfinite grads, the zero-init corrector's final layer
  receives grad on batch 1, hidden layers and the HybridBlock receive grad on
  batch 2 (after the first optimizer step makes the final 1x1 conv nonzero);
* the set of parameters with ``grad is None`` per arm (the reason DDP runs
  need ``find_unused_parameters: true``) matches expectations;
* frozen-prior BatchNorm invariant in ``stable_physical`` mode: running stats
  stay exactly (0, 1) and ``num_batches_tracked`` does not move;
* residual-warmup freeze partitions ``requires_grad`` exactly into
  {corrector, hybrid_block} vs backbone, and restores afterwards;
* humidity round-trip r -> q -> r on real physical fields;
* eval-mode determinism (two identical forwards are bitwise equal);
* val-path behaviour: diagnostics populated under ``torch.no_grad``;
* bf16 autocast forward stays finite (informational; production runs fp32).

The script exits with the number of FAILed checks (0 == everything passed)
and writes a JSON report.

Usage (repo root; the sbatch wrapper sh_files/sanity_pi_iam4vp_1gpu.sh sets
PYTHONPATH and stages the memmap):

    PYTHONPATH=. python tools/sanity_pi_iam4vp_gpu.py \
        --memmap "$WEATHERPRED_USA_MEMMAP" --out logs/sanity_pi_iam4vp.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.normalize import WeatherNormalize
from utils.physics_hybrid import (
    relative_to_specific_humidity,
    specific_to_relative_humidity,
)
from utils.registry import get_model

# ``train.py`` (the entrypoint module) is shadowed by the ``train/`` package on
# sys.path, so its config/dataset factories are loaded here by file path.
# Executing it also registers datasets/models/strategies in utils.registry.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "weatherpred_train_entrypoint", os.path.join(_REPO_ROOT, "train.py")
)
_train_entrypoint = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_train_entrypoint)
build_dataset = _train_entrypoint.build_dataset
load_config = _train_entrypoint.load_config

ARM_CONFIGS: dict[str, str] = {
    "plain_ai": "configs/iam4vp_usa_v4.yaml",
    "legacy_latent": "configs/pi_iam4vp_usa_v4.yaml",
    "fixedeq": "configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml",
    "legacy_hybrid": "configs/pi_iam4vp_residual_legacy_hybrid_usa_v4.yaml",
    "no_physics": "configs/pi_iam4vp_residual_no_physics_usa_v4.yaml",
    "massconsistent": "configs/pi_iam4vp_residual_massconsistent_usa_v4.yaml",
    "diabatic": "configs/pi_iam4vp_residual_diabatic_usa_v4.yaml",
}

TIME_PREDICTION = 6
WARMUP_TRAINABLE_PREFIXES = ("physics_residual_corrector", "hybrid_block")


class CheckLog:
    """Accumulates (name, status, details) rows and counts failures.

    Status semantics: ``PASS``/``FAIL`` are asserted invariants, ``INFO`` is
    observability-only, ``SKIP`` marks checks not applicable in this run.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, arm: str, name: str, status: str, details: object = "") -> None:
        """Append one check result and echo it to stdout."""
        self.rows.append({"arm": arm, "check": name, "status": status, "details": details})
        print(f"[{status:>4}] {arm:>14} :: {name} :: {details}")

    def ok(self, arm: str, name: str, condition: bool, details: object = "") -> None:
        """Add PASS when ``condition`` else FAIL."""
        self.add(arm, name, "PASS" if condition else "FAIL", details)

    @property
    def num_failed(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "FAIL")


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    """Finite-aware summary stats of a tensor (for the JSON report)."""
    finite = torch.isfinite(x)
    total = x.numel()
    if int(finite.sum()) == 0:
        return {
            "nonfinite_ratio": 1.0,
            "min": float("nan"),
            "max": float("nan"),
            "rms": float("nan"),
        }
    xf = x[finite].float()
    return {
        "nonfinite_ratio": float((~finite).float().mean()),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "rms": float(xf.square().mean().sqrt()),
    }


def grad_norm_for_prefix(model: nn.Module, prefix: str) -> float:
    """Sum of |grad| over parameters whose dotted name starts with ``prefix``."""
    total = 0.0
    for name, param in model.named_parameters():
        if name.startswith(prefix) and param.grad is not None:
            total += float(param.grad.detach().abs().sum())
    return total


def none_grad_prefixes(model: nn.Module) -> set[str]:
    """Top-level module prefixes whose parameters all have ``grad is None``."""
    prefix_state: dict[str, list[bool]] = {}
    for name, param in model.named_parameters():
        prefix = name.split(".", 1)[0]
        prefix_state.setdefault(prefix, []).append(param.grad is None)
    return {prefix for prefix, flags in prefix_state.items() if all(flags)}


def count_nonfinite_grads(model: nn.Module) -> int:
    """Number of parameters whose gradient contains a nonfinite value."""
    return sum(
        1
        for param in model.parameters()
        if param.grad is not None and not bool(torch.isfinite(param.grad).all())
    )


def snapshot_hybrid_bn(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Clone (running_mean, running_var, num_batches_tracked) of every hybrid BN."""
    snaps = []
    for module in model.hybrid_block.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            snaps.append(
                (
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone(),
                )
            )
    return snaps


def hybrid_bn_frozen_at_identity(model: nn.Module) -> tuple[bool, float]:
    """Whether every hybrid BN still has running stats (0, 1); returns max drift."""
    max_drift = 0.0
    frozen = True
    for module in model.hybrid_block.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            drift = float(
                torch.maximum(
                    module.running_mean.detach().abs().max(),
                    (module.running_var.detach() - 1.0).abs().max(),
                )
            )
            max_drift = max(max_drift, drift)
            if drift != 0.0:
                frozen = False
    return frozen, max_drift


def build_arm_model(
    config_path: str, device: torch.device, mean: torch.Tensor, std: torch.Tensor
) -> nn.Module:
    """Instantiate one arm exactly as ``train.py`` does and prime normalization."""
    config = load_config(config_path)
    model = get_model(config["model"]["type"])(**config["model"].get("params", {}))
    model = model.to(device)
    if hasattr(model, "set_physics_normalization"):
        model.set_physics_normalization(mean, std)
    return model


def rollout_step_losses(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    backward: bool,
) -> tuple[list[float], list[float], dict[str, float]]:
    """One IterativeManualStep-style rollout over TIME_PREDICTION steps.

    Args:
        model: the arm under test (unwrapped, single GPU).
        x: normalized input ``(B, T, C, H, W)``.
        y: normalized target ``(B, T, C, H, W)``.
        device: CUDA/CPU device of the batch.
        backward: train path (per-step ``backward()``) when True.

    Returns:
        ``(per-step forecast losses, per-step aux losses, mean diagnostics)``.
    """
    step_losses: list[float] = []
    aux_losses: list[float] = []
    diagnostics_sum: dict[str, float] = {}
    pred_list: list[torch.Tensor] = []
    for idx_time in range(TIME_PREDICTION):
        t = torch.tensor((idx_time + 1) * 100, device=device).repeat(x.shape[0])
        prediction = model(x, pred_list, t)
        pred_list.append(prediction.detach())
        forecast_loss = F.l1_loss(prediction, y[:, idx_time])
        aux = model.physics_residual_aux_loss()
        aux_loss = aux if aux is not None else torch.zeros((), device=device)
        if backward:
            (forecast_loss + aux_loss).backward()
        step_losses.append(float(forecast_loss.detach()))
        aux_losses.append(float(aux_loss.detach()))
        for key, value in model.physics_residual_diagnostics().items():
            diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value)
    diagnostics = {key: value / TIME_PREDICTION for key, value in diagnostics_sum.items()}
    return step_losses, aux_losses, diagnostics


def expected_none_grad_prefixes(model: nn.Module) -> set[str]:
    """Parameter prefixes legitimately unused by the arm's forward pass."""
    expected: set[str] = set()
    uses_legacy_latent = model.use_physics and not model.use_physics_residual_corrector
    if not uses_legacy_latent:
        expected.add("lp_phys")
    runs_hybrid = uses_legacy_latent or (
        model.use_physics_residual_corrector and model.physics_feature_mode != "no_physics"
    )
    # stable_physical_v2 runs the HybridBlock as a pure primitive-equation
    # integrator (physics_only_forward), which touches only geometry buffers, so
    # every hybrid_block *parameter* is unused even though the block runs.
    is_v2 = getattr(model, "_hybrid_physical_passthrough", False)
    if not runs_hybrid or is_v2:
        expected.add("hybrid_block")
    return expected


def check_train_rollout(
    log: CheckLog,
    arm: str,
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    lr: float,
) -> dict[str, object]:
    """Two train batches with grad inspection between them; returns report data."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.9), fused=device.type == "cuda"
    )
    has_corrector = model.physics_residual_corrector is not None
    uses_physics_features = (has_corrector and model.physics_feature_mode != "no_physics") or (
        model.use_physics and not has_corrector
    )
    report: dict[str, object] = {}

    bn_before = snapshot_hybrid_bn(model)

    for batch_idx, (x, y) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        step_losses, aux_losses, diagnostics = rollout_step_losses(
            model, x, y, device, backward=True
        )
        losses_finite = all(math.isfinite(value) for value in step_losses + aux_losses)
        diag_finite = all(math.isfinite(value) for value in diagnostics.values())
        log.ok(
            arm,
            f"batch{batch_idx}_losses_finite",
            losses_finite,
            f"forecast={step_losses[0]:.4f}..{step_losses[-1]:.4f} aux={aux_losses[-1]:.3e}",
        )
        log.ok(arm, f"batch{batch_idx}_diagnostics_finite", diag_finite, f"{len(diagnostics)} keys")
        report[f"batch{batch_idx}_step_losses"] = step_losses
        report[f"batch{batch_idx}_aux_losses"] = aux_losses
        report[f"batch{batch_idx}_diagnostics"] = diagnostics

        nonfinite_grads = count_nonfinite_grads(model)
        log.ok(
            arm, f"batch{batch_idx}_grads_finite", nonfinite_grads == 0, f"bad={nonfinite_grads}"
        )

        if batch_idx == 0:
            unused = none_grad_prefixes(model)
            expected_unused = expected_none_grad_prefixes(model)
            log.ok(
                arm,
                "unused_param_prefixes_expected",
                unused == expected_unused,
                f"actual={sorted(unused)} expected={sorted(expected_unused)}",
            )
            report["unused_param_prefixes"] = sorted(unused)
            if has_corrector:
                # Zero-init final 1x1 conv: on batch 0 only net.6 can receive a
                # nonzero grad; hidden convs get exact zeros through W3 == 0.
                final_grad = grad_norm_for_prefix(model, "physics_residual_corrector.net.6")
                log.ok(
                    arm,
                    "corrector_final_layer_grad_nonzero_b0",
                    final_grad > 0.0,
                    f"|g|={final_grad:.3e}",
                )
        if batch_idx == 1:
            if has_corrector:
                # After one optimizer step W3 != 0, so grads must reach net.0.
                hidden_grad = grad_norm_for_prefix(model, "physics_residual_corrector.net.0")
                log.ok(
                    arm,
                    "corrector_hidden_grad_nonzero_b1",
                    hidden_grad > 0.0,
                    f"|g|={hidden_grad:.3e}",
                )
            is_v2 = getattr(model, "_hybrid_physical_passthrough", False)
            if is_v2:
                # v2 is a pure frozen-physics integrator: hybrid_block params must
                # get NO gradient (defect-B fix bypasses conv/norm/router entirely).
                hybrid_grad = grad_norm_for_prefix(model, "hybrid_block")
                log.ok(arm, "hybrid_block_no_grad_v2", hybrid_grad == 0.0, f"|g|={hybrid_grad:.3e}")
            elif uses_physics_features and not model.physics_prior_detach:
                hybrid_grad = grad_norm_for_prefix(model, "hybrid_block")
                log.ok(
                    arm, "hybrid_block_grad_nonzero_b1", hybrid_grad > 0.0, f"|g|={hybrid_grad:.3e}"
                )
            if model.diabatic_head is not None:
                diabatic_grad = grad_norm_for_prefix(model, "diabatic_head")
                log.ok(
                    arm,
                    "diabatic_head_grad_nonzero_b1",
                    diabatic_grad > 0.0,
                    f"|g|={diabatic_grad:.3e}",
                )
        optimizer.step()

    nonfinite_weights = [
        name
        for name, param in model.named_parameters()
        if not bool(torch.isfinite(param.detach()).all())
    ]
    log.ok(
        arm,
        "weights_finite_after_steps",
        not nonfinite_weights,
        f"poisoned={nonfinite_weights[:4]}",
    )
    if has_corrector:
        guard_counts = [
            report[key].get("physics_hybrid_nonfinite_grad_params", 0.0)
            for key in ("batch0_diagnostics", "batch1_diagnostics")
        ]
        log.add(
            arm,
            "hybrid_grad_guard_activations",
            "INFO",
            f"sanitized params per step (b0, b1 diagnostics): {guard_counts}",
        )

    stable_mode = (
        has_corrector
        and model.physics_feature_mode != "no_physics"
        and model.physics_residual_hybrid_mode == "stable_physical"
    )
    frozen, max_drift = hybrid_bn_frozen_at_identity(model)
    bn_after = snapshot_hybrid_bn(model)
    bn_untouched = all(
        torch.equal(before[0], after[0])
        and torch.equal(before[1], after[1])
        and torch.equal(before[2], after[2])
        for before, after in zip(bn_before, bn_after, strict=True)
    )
    if stable_mode:
        log.ok(
            arm,
            "bn_frozen_prior_invariant",
            frozen and bn_untouched,
            f"running stats identity={frozen}, untouched={bn_untouched}, max_drift={max_drift:.3e}",
        )
    else:
        log.add(
            arm,
            "bn_running_stats",
            "INFO",
            f"untouched={bn_untouched}, max_drift_from_identity={max_drift:.3e} (legacy/train-mode BN is expected to move when the hybrid path runs)",
        )
    report["bn_max_drift"] = max_drift
    report["bn_untouched"] = bn_untouched
    return report


def check_warmup_freeze(log: CheckLog, arm: str, model: nn.Module) -> None:
    """set_residual_warmup(True/False) partitions requires_grad correctly."""
    if model.physics_residual_corrector is None:
        log.add(arm, "warmup_freeze", "SKIP", "no residual corrector")
        return
    model.set_residual_warmup(True)
    frozen_violations = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad != name.startswith(WARMUP_TRAINABLE_PREFIXES)
    ]
    model.set_residual_warmup(False)
    unfrozen_violations = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]
    log.ok(
        arm,
        "warmup_freeze_partition",
        not frozen_violations and not unfrozen_violations,
        f"frozen_violations={frozen_violations[:3]} unfrozen_violations={unfrozen_violations[:3]}",
    )


def check_eval_paths(
    log: CheckLog,
    arm: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    """Determinism, val-path diagnostics and bf16 autocast finiteness."""
    report: dict[str, object] = {}
    model.eval()
    t0 = torch.tensor(100, device=device).repeat(x.shape[0])
    with torch.no_grad():
        out_a = model(x, [], t0)
        out_b = model(x, [], t0)
    log.ok(arm, "eval_determinism", torch.equal(out_a, out_b), "two forwards bitwise equal")
    report["eval_output"] = tensor_stats(out_a)

    with torch.no_grad():
        _, aux_losses, diagnostics = rollout_step_losses(model, x, y, device, backward=False)
    diag_finite = all(math.isfinite(value) for value in diagnostics.values())
    if model.physics_residual_corrector is not None:
        log.ok(
            arm,
            "val_path_diagnostics_populated",
            len(diagnostics) > 0 and diag_finite,
            f"{len(diagnostics)} keys, aux_last={aux_losses[-1]:.3e}",
        )
    report["val_diagnostics"] = diagnostics

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_bf16 = model(x, [], t0)
        stats = tensor_stats(out_bf16.float())
        rel = float((out_bf16.float() - out_a).norm() / (out_a.norm() + 1e-12))
        status = "INFO" if stats["nonfinite_ratio"] == 0.0 else "FAIL"
        log.add(arm, "bf16_autocast_forward", status, f"rel_diff={rel:.3e} stats={stats}")
        report["bf16"] = {"rel_diff": rel, **stats}
    return report


def check_humidity_roundtrip(
    log: CheckLog,
    x_raw: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> dict[str, float]:
    """r -> q -> r round-trip on real physical fields from the batch.

    Args:
        x_raw: denormalized physical state ``(B, T, C, H, W)`` (C=69).
        pressure_pa: ``(1, 13, 1, 1)`` pressure levels in Pa.

    Returns:
        Error statistics dict (also logged as PASS/FAIL rows).
    """
    frames = x_raw.reshape(-1, x_raw.shape[2], *x_raw.shape[3:])
    t_kelvin = frames[:, 17:30].clamp(150.0, 350.0)
    r_percent = frames[:, 30:43].clamp(0.0, 150.0)
    q = relative_to_specific_humidity(r_percent, t_kelvin, pressure_pa)
    q_finite = bool(torch.isfinite(q).all())
    q_in_bounds = bool(((q >= 0.0) & (q <= 0.08)).all())
    log.ok("global", "humidity_q_finite", q_finite, tensor_stats(q))
    log.ok("global", "humidity_q_bounds[0,0.08]", q_in_bounds, "")

    r_back = specific_to_relative_humidity(q, t_kelvin, pressure_pa)
    err = (r_back - r_percent).abs()
    unclamped = (q > 1.001e-8) & (q < 0.0799)
    if int(unclamped.sum()) > 0:
        masked_err = err[unclamped]
        max_err = float(masked_err.max())
        mean_err = float(masked_err.mean())
    else:
        max_err, mean_err = 0.0, 0.0
    log.ok(
        "global",
        "humidity_roundtrip_error",
        torch.isfinite(err).all().item() and max_err < 1.0,
        f"unclamped mean={mean_err:.4f}%RH max={max_err:.4f}%RH over {int(unclamped.sum())} pts",
    )
    return {"roundtrip_mean_err": mean_err, "roundtrip_max_err": max_err}


def main() -> None:
    """Entry point: build data once, run every arm, write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap", default=None, help="Path to the packed USA memmap .dat")
    parser.add_argument(
        "--arms", nargs="*", default=sorted(ARM_CONFIGS), choices=sorted(ARM_CONFIGS)
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", default="logs/sanity_pi_iam4vp_gpu.json")
    args = parser.parse_args()

    if args.memmap:
        os.environ["MEMMAP_PATH_OVERRIDE"] = args.memmap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    print(f"[sanity] device={device} torch={torch.__version__}")

    base_config = load_config(ARM_CONFIGS["fixedeq"])
    dataset = build_dataset(base_config["data"], "val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    normalize = WeatherNormalize(
        mean=torch.as_tensor(dataset.the_mean, dtype=torch.float32),
        std=torch.as_tensor(dataset.the_std, dtype=torch.float32),
    ).to(device)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    raw_first: tuple[torch.Tensor, torch.Tensor] | None = None
    for x_raw, y_raw in loader:
        x_raw, y_raw = x_raw.to(device), y_raw.to(device)
        if raw_first is None:
            raw_first = (x_raw, y_raw)
        batches.append((normalize(x_raw), normalize(y_raw)))
        if len(batches) == 2:
            break
    if len(batches) < 2:
        print("[sanity] FATAL: dataset yielded fewer than 2 batches")
        sys.exit(2)
    print(f"[sanity] data ready: x{tuple(batches[0][0].shape)} y{tuple(batches[0][1].shape)}")

    log = CheckLog()
    report: dict[str, object] = {"device": str(device), "torch": torch.__version__, "arms": {}}

    diabatic_config = load_config(ARM_CONFIGS["diabatic"])
    diabatic_constants = diabatic_config["model"]["params"].get("diabatic_constants_path")
    arms = list(args.arms)
    if "diabatic" in arms and (not diabatic_constants or not os.path.exists(diabatic_constants)):
        log.add("diabatic", "arm", "SKIP", f"constants not found: {diabatic_constants}")
        arms.remove("diabatic")

    for arm in arms:
        t_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_arm_model(
            ARM_CONFIGS[arm], device, dataset.data_mean_tensor, dataset.data_std_tensor
        )
        arm_report: dict[str, object] = {
            "config": ARM_CONFIGS[arm],
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        arm_report["train"] = check_train_rollout(log, arm, model, batches, device, args.lr)
        check_warmup_freeze(log, arm, model)
        arm_report["eval"] = check_eval_paths(log, arm, model, batches[0][0], batches[0][1], device)
        arm_report["wall_seconds"] = round(time.time() - t_start, 1)
        if device.type == "cuda":
            arm_report["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["arms"][arm] = arm_report  # type: ignore[index]
        print(
            f"[sanity] arm={arm} done in {arm_report['wall_seconds']}s "
            f"mem={arm_report.get('max_mem_gib', 'n/a')}GiB"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixedeq_model = build_arm_model(
        ARM_CONFIGS["fixedeq"], device, dataset.data_mean_tensor, dataset.data_std_tensor
    )
    x_first_raw = raw_first[0]
    report["humidity"] = check_humidity_roundtrip(
        log, x_first_raw, fixedeq_model.physics_pressure_pa
    )
    del fixedeq_model

    report["checks"] = log.rows
    report["num_failed"] = log.num_failed
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[sanity] report written to {args.out}")

    verdict = "OK" if log.num_failed == 0 else f"FAIL ({log.num_failed} checks)"
    print(f"[sanity] SANITY {verdict}: {len(log.rows)} checks total")
    sys.exit(min(log.num_failed, 125))


if __name__ == "__main__":
    main()
