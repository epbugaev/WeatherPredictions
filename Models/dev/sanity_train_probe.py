"""Single-card training probe for the PredFormer-USA v4 setup.

Cheap go/no-go BEFORE a 6-day 2-GPU DDP run: drives the *real* per-batch path
(PredFormer + ``WeatherNormalize`` + ``SimpleStep`` + AdamW + ``Metrics.WRMSE``)
for a few hundred optimiser steps on a fixed synthetic batch, then reports:

  * loss trajectory + finiteness (does it explode / is the loss-scale sane?),
  * train/val gap on a held-out synthetic batch (overfit-a-batch sanity),
  * per-variable lat-weighted RMSE for **all 69 channels** (the Pareto metric),
    so a config change can be smoke-checked per channel before the cluster run.

It does NOT replace a real long run for judging overfit — it catches pipeline
regressions, NaN/Inf, and gross loss-scale shifts (e.g. MAE -> MSE) in minutes.

Run on the cluster (where the constants netCDF lives), single card::

    python -m Models.dev.sanity_train_probe \
        --path-to-constants \
        /home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc \
        --steps 300 --loss-type MSE --weight-decay 0.02

``--ndepth`` shrinks the 24-block stack for a fast CPU smoke (default 24 mirrors
the production config). mean/std come from ``example_data/mean_std.json`` when
present (realistic per-channel scale), else identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Models.PredFormer import PredFormer_Model
from training_strategies.base import StepContext
from training_strategies.simple import SimpleStep
from utils.metrics import Metrics
from utils.normalize import WeatherNormalize

REPO_ROOT = Path(__file__).resolve().parents[2]
MEAN_STD_PATH = REPO_ROOT / "example_data" / "mean_std.json"
NUM_CHANNELS = 69


def load_mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel ``(mean, std)`` of shape ``(69,)``.

    Reads ``example_data/mean_std.json`` when it exists (realistic scale);
    otherwise falls back to ``mean=0, std=1`` (normalisation becomes identity,
    still valid for a pipeline / loss-scale sanity).

    Returns:
        Tuple ``(mean, std)`` float32 tensors of shape ``(69,)``.
    """
    if MEAN_STD_PATH.exists():
        stats = json.loads(MEAN_STD_PATH.read_text())
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        return mean, std
    return torch.zeros(NUM_CHANNELS), torch.ones(NUM_CHANNELS)


def build_model_config(path_to_constants: str, ndepth: int, drop_path: float) -> dict:
    """Return a PredFormer model_config mirroring ``predformer_usa_v4.yaml``."""
    return {
        "height": 32,
        "width": 64,
        "cut": [[75, 107], [164, 228]],
        "num_channels": NUM_CHANNELS,
        "pre_seq": 12,
        "after_seq": 12,
        "patch_size": 8,
        "dim": 256,
        "heads": 8,
        "dim_head": 32,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "drop_path": drop_path,
        "scale_dim": 4,
        "depth": 1,
        "Ndepth": ndepth,
        "path_to_constants": path_to_constants,
    }


def make_batch(
    batch: int,
    normalize: WeatherNormalize,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize one normalised ``(x, y)`` pair of shape ``(B, 12, 69, 32, 64)``.

    Draws raw physical-scale samples ``mean + std * randn`` then applies the
    trainer's ``WeatherNormalize`` (so the post-norm distribution is ~N(0,1),
    matching what the model/loss actually see at train time).
    """
    shape = (batch, 12, NUM_CHANNELS, 32, 64)
    scale = std.view(1, 1, -1, 1, 1)
    shift = mean.view(1, 1, -1, 1, 1)
    x_raw = shift + scale * torch.randn(shape)
    y_raw = shift + scale * torch.randn(shape)
    x = normalize(x_raw.to(device))
    y = normalize(y_raw.to(device))
    return x, y


def report_per_channel_rmse(val_metrics: dict[str, torch.Tensor]) -> bool:
    """Print worst-10 per-channel RMSE (last timestep) and return all-finite.

    Args:
        val_metrics: dict from ``SimpleStep.val_step`` (``RMSE_<var>_first/last``).

    Returns:
        ``True`` if every reported RMSE is finite.
    """
    rows = [
        (name.removeprefix("RMSE_").removesuffix("_last"), float(value))
        for name, value in val_metrics.items()
        if name.endswith("_last")
    ]
    all_finite = all(torch.isfinite(torch.tensor(v)) for _, v in rows)
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"  per-channel RMSE_last: {len(rows)}/{NUM_CHANNELS} channels logged")
    print("  worst 10: " + ", ".join(f"{n}={v:.4f}" for n, v in rows[:10]))
    print(f"  all per-channel RMSE finite: {all_finite}")
    return all_finite


def main() -> None:
    """Parse args, run the probe, raise on any non-finite signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-to-constants", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=("MAE", "MSE"), default="MAE")
    parser.add_argument("--drop-path", type=float, default=0.15)
    parser.add_argument("--ndepth", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device("cuda", 0)
        if (not args.cpu and torch.cuda.is_available())
        else torch.device("cpu")
    )
    print(
        f"Device: {device} | loss={args.loss_type} wd={args.weight_decay} "
        f"lr={args.lr} steps={args.steps} batch={args.batch} Ndepth={args.ndepth}"
    )

    mean, std = load_mean_std()
    normalize = WeatherNormalize(mean=mean, std=std).to(device)
    metrics = Metrics(mean.to(device), std.to(device))
    strategy = SimpleStep(loss_type=args.loss_type)

    model = PredFormer_Model(
        build_model_config(args.path_to_constants, args.ndepth, args.drop_path)
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.9)
    )
    ctx = StepContext(
        device=device,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler(enabled=False),
        experiment=None,
        metrics=metrics,
        global_step=0,
        epoch=0,
        is_main_process=True,
    )

    x_train, y_train = make_batch(args.batch, normalize, mean, std, device)
    x_val, y_val = make_batch(args.batch, normalize, mean, std, device)

    model.train()
    first_loss = None
    last_loss = None
    for step in range(args.steps):
        step_metrics = strategy.train_step(model, (x_train, y_train), ctx)
        loss = step_metrics["loss"]
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite train loss at step {step}: {loss.item()!r}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        if first_loss is None:
            first_loss = last_loss
        if step % max(1, args.steps // 10) == 0:
            print(f"  step {step:4d}  train_loss={last_loss:.5f}")

    model.eval()
    with torch.no_grad():
        val_metrics = strategy.val_step(model, (x_val, y_val), ctx)
    val_loss = float(val_metrics["val_loss"])

    print(f"\ntrain_loss: {first_loss:.5f} -> {last_loss:.5f} (drop {first_loss - last_loss:+.5f})")
    print(f"val_loss (held-out batch): {val_loss:.5f}  | gap={val_loss - last_loss:+.5f}")
    all_finite = report_per_channel_rmse(val_metrics)

    if not all_finite or not torch.isfinite(torch.tensor(val_loss)):
        raise AssertionError("probe produced non-finite metrics — pipeline regression")
    if last_loss >= first_loss:
        raise AssertionError(
            f"train loss did not decrease ({first_loss:.5f} -> {last_loss:.5f}); "
            "loss-scale / lr likely broken for this config"
        )
    print("\nProbe PASSED: finite, loss decreasing, per-channel metrics computed.")


if __name__ == "__main__":
    main()
