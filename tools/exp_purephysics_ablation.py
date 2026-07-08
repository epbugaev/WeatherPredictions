"""Before/after ablation of the defect-B fix (stable_physical -> stable_physical_v2).

Both conditions are built from the SAME arm config and seed on the current code
tree, differing only in ``physics_residual_hybrid_mode``: ``stable_physical``
(the contaminated baseline — the random ``variable_norm`` conv smears the
z-scale across every block) vs ``stable_physical_v2`` (pure primitive-equation
integration of the clean physical state). Runs K optimizer steps and logs, per
step, whether the physics feature is physical: the forward nonfinite ratio, how
often the tendency saturates its clip, the tendency RMS, the physical
temperature range of the prior, and the task loss.

Usage:
    PYTHONPATH=. python tools/exp_purephysics_ablation.py \
        --memmap "$MEMMAP" --config configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml \
        --steps 12 --out logs/purephys.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.normalize import WeatherNormalize
from utils.registry import get_model

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "weatherpred_train_entrypoint", os.path.join(_REPO_ROOT, "train.py")
)
_train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_train)

TIME_PREDICTION = 6
MODES = ("stable_physical", "stable_physical_v2")


def feature_health(model: nn.Module, prior_t_kelvin_range: tuple[float, float]) -> dict[str, float]:
    """Physics-feature quality metrics after one rollout step."""
    diagnostics = model.physics_residual_diagnostics()

    def diag(key: str) -> float:
        value = diagnostics.get(key)
        return float(value) if value is not None else float("nan")

    forward_ratio = model._last_physics_nonfinite_ratio
    return {
        "forward_nonfinite_ratio": (
            float(forward_ratio) if forward_ratio is not None else float("nan")
        ),
        "tendency_clip_ratio": diag("physics_residual_tendency_clip_ratio"),
        "physics_tendency_rms": diag("physics_residual_tendency_rms"),
        "correction_to_tendency_cosine": diag("physics_residual_correction_to_tendency_cosine"),
        "prior_t_min_kelvin": prior_t_kelvin_range[0],
        "prior_t_max_kelvin": prior_t_kelvin_range[1],
    }


def prior_t_range(model: nn.Module, prev_state: torch.Tensor) -> tuple[float, float]:
    """Physical temperature range (K) of the physics prior for the given state."""
    with torch.no_grad():
        prior = model._physics_prior_from_state(prev_state)
        t_block = model._denormalize_state(prior)[:, 17:30]
    return float(t_block.min()), float(t_block.max())


def run_mode(
    mode: str,
    config: dict,
    dataset,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Build the arm in ``mode`` and log per-step physics-feature health."""
    torch.manual_seed(seed)
    params = dict(config["model"].get("params", {}))
    params["physics_residual_hybrid_mode"] = mode
    model = get_model(config["model"]["type"])(**params).to(device)
    model.set_physics_normalization(dataset.data_mean_tensor, dataset.data_std_tensor)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9))

    log: list[dict[str, float]] = []
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
        t_range = prior_t_range(model, x[:, -1])
        health = feature_health(model, t_range)
        health["step"] = step
        health["task_loss"] = task_loss / TIME_PREDICTION
        log.append(health)
    return log


def main() -> None:
    """Entry point: run both modes and write the comparison JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap", default=None)
    parser.add_argument("--config", default="configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.memmap:
        os.environ["MEMMAP_PATH_OVERRIDE"] = os.path.abspath(args.memmap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[purephys] device={device} config={args.config}")

    config = _train.load_config(args.config)
    dataset = _train.build_dataset(config["data"], "val")
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

    report: dict[str, object] = {"config": args.config, "modes": {}}
    for mode in MODES:
        log = run_mode(mode, config, dataset, batches, args.steps, args.lr, args.seed, device)
        report["modes"][mode] = log  # type: ignore[index]
        last = log[-1]
        print(
            f"[purephys] {mode:20s}: nf={last['forward_nonfinite_ratio']:.3f} "
            f"clip={last['tendency_clip_ratio']:.3f} tend_rms={last['physics_tendency_rms']:.3e} "
            f"t=[{last['prior_t_min_kelvin']:.0f},{last['prior_t_max_kelvin']:.0f}]K "
            f"loss={last['task_loss']:.4f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[purephys] written to {args.out}")


if __name__ == "__main__":
    main()
