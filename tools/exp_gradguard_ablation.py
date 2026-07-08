"""Before/after ablation of the HybridBlock NaN-gradient guard (defect A).

Runs K optimizer steps of ``stable_physical`` PI-IAM4VP arms and logs, per
step, the health of the physics branch: whether ``hybrid_block`` weights have
gone nonfinite (the poisoning), the forward nonfinite ratio, whether the router
is alive, and how much physics-tendency signal survives.

Method — literal old-code vs new-code. ``--repo`` selects the code tree the
model / dataset / config factories are imported from, so the same harness runs
against the pre-fix worktree (commit 4d2dea8) and the post-fix tree
(6c4a645+). A fixed ``--seed`` makes model initialisation bit-identical across
the two trees (verified by the state_dict/forward harness), and the batches are
read deterministically from the same memmap, so the registered gradient guard
is the *only* difference between the two runs. Run the harness twice (one
``--repo`` each) and diff the JSONs.

Usage:
    # after (current tree)
    PYTHONPATH=. python tools/exp_gradguard_ablation.py \
        --repo . --label after --memmap "$MEMMAP" \
        --arms fixedeq massconsistent legacy_hybrid --steps 12 \
        --out logs/gg_after.json
    # before (pre-fix worktree at 4d2dea8)
    PYTHONPATH=<worktree> python tools/exp_gradguard_ablation.py \
        --repo <worktree> --label before --memmap "$MEMMAP" \
        --arms fixedeq massconsistent legacy_hybrid --steps 12 \
        --out logs/gg_before.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ARM_CONFIGS: dict[str, str] = {
    "fixedeq": "configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml",
    "massconsistent": "configs/pi_iam4vp_residual_massconsistent_usa_v4.yaml",
    "legacy_hybrid": "configs/pi_iam4vp_residual_legacy_hybrid_usa_v4.yaml",
    "no_physics": "configs/pi_iam4vp_residual_no_physics_usa_v4.yaml",
}

TIME_PREDICTION = 6


def load_repo_modules(repo: str):
    """Import the model/train factories from ``repo`` and return them.

    Inserts ``repo`` at the front of ``sys.path`` and loads ``train.py`` by
    file path (the ``train/`` package shadows the module name). Executing
    ``train.py`` also registers datasets/models/strategies in the repo's
    ``utils.registry``.

    Args:
        repo: path to the repository root (or git worktree) to import from.

    Returns:
        Tuple ``(train_entrypoint_module, get_model, WeatherNormalize)``.
    """
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    registry = importlib.import_module("utils.registry")
    normalize_mod = importlib.import_module("utils.normalize")
    spec = importlib.util.spec_from_file_location(
        "weatherpred_train_entrypoint", os.path.join(repo, "train.py")
    )
    train_entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_entrypoint)
    return train_entrypoint, registry.get_model, normalize_mod.WeatherNormalize


def physics_health(model: torch.nn.Module) -> dict[str, object]:
    """Snapshot the physics-branch health metrics of ``model``.

    Reads directly from parameters and the model's own diagnostics dict, using
    only keys/attributes that exist on both sides of the fix, so the same
    function runs against the pre-fix and post-fix trees.

    Args:
        model: an IAM4VP instance after a forward/backward/step.

    Returns:
        Dict of scalar health metrics (JSON-serialisable).
    """
    hybrid = model.hybrid_block
    nonfinite_params = [
        name for name, param in hybrid.named_parameters() if not bool(torch.isfinite(param).all())
    ]
    router = hybrid.router_weight.detach()
    router_finite = bool(torch.isfinite(router).all())
    diagnostics = model.physics_residual_diagnostics()

    def diag(key: str) -> float:
        value = diagnostics.get(key)
        return float(value) if value is not None else float("nan")

    forward_ratio = model._last_physics_nonfinite_ratio
    return {
        "hybrid_nonfinite_param_count": len(nonfinite_params),
        "hybrid_nonfinite_params": nonfinite_params[:6],
        "router_finite": router_finite,
        "router_abs_mean": float(router.abs().mean()) if router_finite else float("nan"),
        "forward_nonfinite_ratio": (
            float(forward_ratio) if forward_ratio is not None else float("nan")
        ),
        "physics_tendency_rms": diag("physics_residual_tendency_rms"),
        "correction_rms": diag("physics_residual_correction_rms"),
        "pi_minus_iam4vp_rms": diag("physics_residual_pi_minus_iam4vp_rms"),
    }


def train_k_steps(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, object]]:
    """Run ``steps`` IterativeManualStep-style updates, logging health per step.

    Args:
        model: the arm under test (single process, unwrapped).
        batches: cyclic pool of ``(x, y)`` normalized batches.
        steps: number of optimizer steps.
        lr: AdamW learning rate (betas 0.9/0.9 matches production).
        device: batch/model device.

    Returns:
        Per-step list of ``physics_health`` dicts augmented with ``step`` and
        ``task_loss`` (mean per-timestep L1).
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9))
    log: list[dict[str, object]] = []
    for step in range(steps):
        x, y = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        pred_list: list[torch.Tensor] = []
        task_loss = 0.0
        for idx_time in range(TIME_PREDICTION):
            t = torch.tensor((idx_time + 1) * 100, device=device).repeat(x.shape[0])
            prediction = model(x, pred_list, t)
            pred_list.append(prediction.detach())
            forecast_loss = F.l1_loss(prediction, y[:, idx_time])
            aux = model.physics_residual_aux_loss()
            (forecast_loss + (aux if aux is not None else 0.0)).backward()
            task_loss += float(forecast_loss.detach())
        optimizer.step()
        health = physics_health(model)
        health["step"] = step
        health["task_loss"] = task_loss / TIME_PREDICTION
        log.append(health)
    return log


def main() -> None:
    """Entry point: run each requested arm for K steps and write the JSON log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="Repo root / worktree to import model code from"
    )
    parser.add_argument("--label", required=True, help="Condition label (e.g. before / after)")
    parser.add_argument("--memmap", default=None, help="Packed USA memmap .dat path")
    parser.add_argument("--arms", nargs="*", default=["fixedeq", "massconsistent", "legacy_hybrid"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    if args.memmap:
        os.environ["MEMMAP_PATH_OVERRIDE"] = os.path.abspath(args.memmap)

    train_entrypoint, get_model, WeatherNormalize = load_repo_modules(args.repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gg-ablation] repo={args.repo} label={args.label} device={device}")

    base_config = train_entrypoint.load_config(ARM_CONFIGS["fixedeq"])
    dataset = train_entrypoint.build_dataset(base_config["data"], "val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    normalize = WeatherNormalize(
        mean=torch.as_tensor(dataset.the_mean, dtype=torch.float32),
        std=torch.as_tensor(dataset.the_std, dtype=torch.float32),
    ).to(device)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for x_raw, y_raw in loader:
        batches.append((normalize(x_raw.to(device)), normalize(y_raw.to(device))))
        if len(batches) == args.num_batches:
            break
    print(f"[gg-ablation] {len(batches)} batches x{tuple(batches[0][0].shape)}")

    report: dict[str, object] = {
        "label": args.label,
        "repo": os.path.abspath(args.repo),
        "arms": {},
    }
    for arm in args.arms:
        torch.manual_seed(args.seed)
        config = train_entrypoint.load_config(ARM_CONFIGS[arm])
        model = get_model(config["model"]["type"])(**config["model"].get("params", {})).to(device)
        if hasattr(model, "set_physics_normalization"):
            model.set_physics_normalization(dataset.data_mean_tensor, dataset.data_std_tensor)
        log = train_k_steps(model, batches, args.steps, args.lr, device)
        report["arms"][arm] = log  # type: ignore[index]
        first, last = log[0], log[-1]
        print(
            f"[gg-ablation] {args.label}/{arm}: "
            f"step0 router_finite={first['router_finite']} nf={first['forward_nonfinite_ratio']:.3f} "
            f"| stepN router_finite={last['router_finite']} nf={last['forward_nonfinite_ratio']:.3f} "
            f"poisoned_params={last['hybrid_nonfinite_param_count']} "
            f"tendency_rms={last['physics_tendency_rms']:.3e} loss={last['task_loss']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[gg-ablation] written to {out_path}")


if __name__ == "__main__":
    main()
