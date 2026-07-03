#!/usr/bin/env python3
"""Train a tiny matrix calibrator on top of equations-only PurePDEKernel.

This experiment keeps the physics path deliberately small:

    PurePDEKernel RHS tendencies
        -> fixed per-variable/level normalization
        -> trainable identity-initialized matrix
        -> denormalization
        -> Euler update

There are no Conv2d blocks, spatial kernels, HybridBlock adapters, residual
heads, or neural backbone. The trainable part is only a small matrix that asks:
"are the PDE tendencies useful if we let ERA5 calibrate their couplings?"

Default matrix mode is ``per_level``: one 5x5 matrix per pressure level for
``z, t, q, u, v`` tendencies. That is 325 trainable parameters and no spatial
mixing. The matrix operates in normalized tendency space, so all coefficients
are dimensionless and comparable across variables.

Use ``--training-mode tendency`` to avoid autoregressive rollout entirely: each
lead is trained as a teacher-forced one-hour correction from the true previous
frame. The older rollout experiment remains available as
``--training-mode autoregressive``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_pure_pde_operator import (  # noqa: E402
    PDE_VARIABLES,
    _to_device,
    _to_physical_batch,
    build_dataset,
    build_kernel,
    physical_state_from_frame,
    pde_step,
    weighted_mse_per_channel,
)
from utils.normalize import WeatherNormalize  # noqa: E402


class TendencyMatrixCorrector(nn.Module):
    """Dimensionless matrix correction for PurePDEKernel RHS tendencies."""

    def __init__(
        self,
        scales: torch.Tensor,
        mode: str = "per_level",
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if scales.shape != (5, 13):
            raise ValueError(f"Expected scales shape (5,13), got {tuple(scales.shape)}")
        if mode not in ("diagonal", "per_level", "full"):
            raise ValueError(f"Unknown matrix mode {mode!r}")
        self.mode = mode
        self.max_delta = float(max_delta)
        self.register_buffer("scales", scales.detach().clone().float())
        if mode == "diagonal":
            self.raw = nn.Parameter(torch.zeros(5, 13))
        elif mode == "per_level":
            self.raw = nn.Parameter(torch.zeros(13, 5, 5))
        else:
            self.raw = nn.Parameter(torch.zeros(65, 65))

    def effective_matrix(self) -> torch.Tensor:
        delta = self.max_delta * torch.tanh(self.raw)
        if self.mode == "diagonal":
            return 1.0 + delta
        if self.mode == "per_level":
            eye = torch.eye(5, device=delta.device, dtype=delta.dtype).view(1, 5, 5)
            return eye + delta
        eye = torch.eye(65, device=delta.device, dtype=delta.dtype)
        return eye + delta

    def identity_penalty(self) -> torch.Tensor:
        matrix = self.effective_matrix()
        if self.mode == "diagonal":
            return (matrix - 1.0).square().mean()
        if self.mode == "per_level":
            eye = torch.eye(5, device=matrix.device, dtype=matrix.dtype).view(1, 5, 5)
            return (matrix - eye).square().mean()
        eye = torch.eye(65, device=matrix.device, dtype=matrix.dtype)
        return (matrix - eye).square().mean()

    def forward(self, rhs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tendencies = torch.stack([rhs[f"{name}_t"] for name in PDE_VARIABLES], dim=1)
        scale = self.scales.to(device=tendencies.device, dtype=tendencies.dtype).view(1, 5, 13, 1, 1)
        normalized = tendencies / scale.clamp_min(1e-12)
        matrix = self.effective_matrix().to(dtype=tendencies.dtype)

        if self.mode == "diagonal":
            mixed = normalized * matrix.view(1, 5, 13, 1, 1)
        elif self.mode == "per_level":
            mixed = torch.einsum("poi,biphw->bophw", matrix, normalized)
        else:
            batch, _, _, height, width = normalized.shape
            flat = normalized.reshape(batch, 65, height, width)
            mixed = torch.einsum("oi,bihw->bohw", matrix, flat).reshape(batch, 5, 13, height, width)

        corrected = mixed * scale
        return {name: corrected[:, idx] for idx, name in enumerate(PDE_VARIABLES)}


def make_normalizer(dataset, device: torch.device) -> WeatherNormalize:
    return WeatherNormalize(
        mean=torch.as_tensor(dataset.the_mean, dtype=torch.float32),
        std=torch.as_tensor(dataset.the_std, dtype=torch.float32),
    ).to(device)


def make_scales(dataset, q_scale: float, device: torch.device) -> torch.Tensor:
    data_std = torch.as_tensor(dataset.data_std_tensor, dtype=torch.float32, device=device)
    scales = torch.stack(
        [
            data_std[4:17],
            data_std[17:30],
            torch.full((13,), float(q_scale), dtype=torch.float32, device=device),
            data_std[43:56],
            data_std[56:69],
        ],
        dim=0,
    )
    return scales.clamp_min(1e-12)


def make_loader(config: dict[str, Any], split: str, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = build_dataset(config["data"], split)
    split_cfg = config["data"].get(split, {})
    batch_size = args.batch_size or split_cfg.get("batch_size", 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )


def stack_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([state[name] for name in PDE_VARIABLES], dim=1)


def normalized_mae(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    scales: torch.Tensor,
) -> torch.Tensor:
    scale = scales.to(device=next(iter(pred.values())).device).view(1, 5, 13, 1, 1)
    return ((stack_state(pred) - stack_state(target)).abs() / scale).mean()


def corrected_euler_step(
    kernel,
    corrector: TendencyMatrixCorrector,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rhs = kernel.rhs(state["u"], state["v"], state["t"], state["q"], state["z"])
    corrected_rhs = corrector(rhs)
    dt = kernel.block_dt
    out = {
        name: state[name] + dt * corrected_rhs[name]
        for name in PDE_VARIABLES
    }
    out.update({f"{name}_t": corrected_rhs[name] for name in PDE_VARIABLES})
    finalized = kernel._finalize(out)
    return {name: finalized[name] for name in PDE_VARIABLES}


def rollout_one_frame(
    kernel,
    corrector: TendencyMatrixCorrector,
    state: dict[str, torch.Tensor],
    substeps: int,
) -> dict[str, torch.Tensor]:
    cur = state
    for _ in range(substeps):
        cur = corrected_euler_step(kernel, corrector, cur)
    return cur


def frame_dt_seconds(args: argparse.Namespace) -> float:
    return float(args.block_dt) * float(args.substeps_per_frame)


def assert_finite_state(state: dict[str, torch.Tensor], label: str) -> None:
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            finite = torch.isfinite(tensor)
            if finite.any():
                finite_values = tensor[finite]
                min_value = float(finite_values.min().detach().cpu())
                max_value = float(finite_values.max().detach().cpu())
            else:
                min_value = float("nan")
                max_value = float("nan")
            raise FloatingPointError(
                f"{label}.{name} contains non-finite values "
                f"(finite_min={min_value:.6g}, finite_max={max_value:.6g})"
            )


def direct_euler_from_rhs(
    kernel,
    state: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    dt: float,
) -> dict[str, torch.Tensor]:
    out = {
        name: state[name] + dt * rhs[f"{name}_t"]
        for name in PDE_VARIABLES
    }
    out.update({f"{name}_t": rhs[f"{name}_t"] for name in PDE_VARIABLES})
    finalized = kernel._finalize(out)
    return {name: finalized[name] for name in PDE_VARIABLES}


def direct_one_step_predictions(
    kernel,
    corrector: TendencyMatrixCorrector,
    state: dict[str, torch.Tensor],
    dt: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rhs = kernel.rhs(state["u"], state["v"], state["t"], state["q"], state["z"])
    corrected_rhs = corrector(rhs)
    matrix_state = direct_euler_from_rhs(
        kernel,
        state,
        {f"{name}_t": corrected_rhs[name] for name in PDE_VARIABLES},
        dt,
    )
    pure_state = direct_euler_from_rhs(kernel, state, rhs, dt)
    return matrix_state, pure_state


def run_train_epoch(
    loader: DataLoader,
    kernel,
    corrector: TendencyMatrixCorrector,
    normalize: WeatherNormalize,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    returns_normalized: bool,
    scales: torch.Tensor,
) -> dict[str, float]:
    corrector.train()
    total_loss = 0.0
    total_data = 0.0
    total_penalty = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if args.train_max_batches is not None and batch_idx >= args.train_max_batches:
            break
        x, y = _to_physical_batch(_to_device(batch, torch.device(args.device)), normalize, returns_normalized)
        horizon = min(args.horizon or y.shape[1], y.shape[1])
        state = physical_state_from_frame(x[:, -1], kernel, args.humidity_mode)
        data_loss = torch.zeros((), device=x.device)

        for lead_idx in range(horizon):
            if args.training_mode == "tendency":
                source_frame = x[:, -1] if lead_idx == 0 else y[:, lead_idx - 1]
                state = physical_state_from_frame(source_frame, kernel, args.humidity_mode)
                state, _ = direct_one_step_predictions(
                    kernel,
                    corrector,
                    state,
                    frame_dt_seconds(args),
                )
            else:
                state = rollout_one_frame(kernel, corrector, state, args.substeps_per_frame)
            assert_finite_state(state, f"train batch={batch_idx} lead={lead_idx + 1} matrix_state")
            target = physical_state_from_frame(y[:, lead_idx], kernel, args.humidity_mode)
            data_loss = data_loss + normalized_mae(state, target, scales)

        data_loss = data_loss / horizon
        penalty = corrector.identity_penalty()
        loss = data_loss + args.identity_lambda * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"train batch={batch_idx} produced non-finite loss "
                f"(data_loss={float(data_loss.detach().cpu())}, "
                f"identity_penalty={float(penalty.detach().cpu())})"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corrector.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_data += float(data_loss.detach().cpu())
        total_penalty += float(penalty.detach().cpu())
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No train batches processed")
    return {
        "train_loss": total_loss / total_batches,
        "train_data_loss": total_data / total_batches,
        "train_identity_penalty": total_penalty / total_batches,
    }


def add_eval_metric(
    stats: dict[tuple[str, int], dict[str, torch.Tensor | int]],
    variable: str,
    lead: int,
    matrix_pred: torch.Tensor,
    pure_pred: torch.Tensor,
    constant_pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
) -> None:
    matrix_mse = weighted_mse_per_channel(matrix_pred, target, lat_weights)
    pure_mse = weighted_mse_per_channel(pure_pred, target, lat_weights)
    constant_mse = weighted_mse_per_channel(constant_pred, target, lat_weights)
    key = (variable, lead)
    if key not in stats:
        stats[key] = {
            "matrix_sum_mse": torch.zeros(matrix_mse.shape[1], device=matrix_mse.device),
            "pure_sum_mse": torch.zeros(pure_mse.shape[1], device=pure_mse.device),
            "constant_sum_mse": torch.zeros(constant_mse.shape[1], device=constant_mse.device),
            "count": 0,
        }
    stats[key]["matrix_sum_mse"] = stats[key]["matrix_sum_mse"] + matrix_mse.sum(dim=0)
    stats[key]["pure_sum_mse"] = stats[key]["pure_sum_mse"] + pure_mse.sum(dim=0)
    stats[key]["constant_sum_mse"] = stats[key]["constant_sum_mse"] + constant_mse.sum(dim=0)
    stats[key]["count"] = int(stats[key]["count"]) + matrix_mse.shape[0]


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    kernel,
    corrector: TendencyMatrixCorrector,
    normalize: WeatherNormalize,
    args: argparse.Namespace,
    returns_normalized: bool,
    max_batches: int | None,
) -> list[dict[str, float | int | str]]:
    corrector.eval()
    stats: dict[tuple[str, int], dict[str, torch.Tensor | int]] = {}
    lat_weights = kernel.grid.config.H * torch.cos(kernel.grid.latitudes) / torch.cos(kernel.grid.latitudes).sum()
    horizon_seen: int | None = None
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = _to_physical_batch(_to_device(batch, torch.device(args.device)), normalize, returns_normalized)
        horizon = min(args.horizon or y.shape[1], y.shape[1])
        if horizon_seen is None:
            horizon_seen = horizon
        elif horizon_seen != horizon:
            raise ValueError("Variable horizon across batches is not supported")

        matrix_state = physical_state_from_frame(x[:, -1], kernel, args.humidity_mode)
        pure_state = {key: value.clone() for key, value in matrix_state.items()}
        constant_state = {key: value.clone() for key, value in matrix_state.items()}
        for lead_idx in range(horizon):
            if args.training_mode == "tendency":
                source_frame = x[:, -1] if lead_idx == 0 else y[:, lead_idx - 1]
                matrix_state = physical_state_from_frame(source_frame, kernel, args.humidity_mode)
                constant_state = {key: value.clone() for key, value in matrix_state.items()}
                matrix_state, pure_state = direct_one_step_predictions(
                    kernel,
                    corrector,
                    matrix_state,
                    frame_dt_seconds(args),
                )
            else:
                matrix_state = rollout_one_frame(kernel, corrector, matrix_state, args.substeps_per_frame)
                pure_state = pde_step(kernel, pure_state, args.substeps_per_frame)
            assert_finite_state(matrix_state, f"val batch={batch_idx} lead={lead_idx + 1} matrix_state")
            assert_finite_state(pure_state, f"val batch={batch_idx} lead={lead_idx + 1} pure_state")
            target_state = physical_state_from_frame(y[:, lead_idx], kernel, args.humidity_mode)
            for variable in PDE_VARIABLES:
                add_eval_metric(
                    stats,
                    variable,
                    lead_idx + 1,
                    matrix_state[variable],
                    pure_state[variable],
                    constant_state[variable],
                    target_state[variable],
                    lat_weights,
                )
            add_eval_metric(
                stats,
                "physics_all",
                lead_idx + 1,
                torch.cat([matrix_state[variable] for variable in PDE_VARIABLES], dim=1),
                torch.cat([pure_state[variable] for variable in PDE_VARIABLES], dim=1),
                torch.cat([constant_state[variable] for variable in PDE_VARIABLES], dim=1),
                torch.cat([target_state[variable] for variable in PDE_VARIABLES], dim=1),
                lat_weights,
            )
        total_batches += 1

    if total_batches == 0 or horizon_seen is None:
        raise RuntimeError("No eval batches processed")

    rows: list[dict[str, float | int | str]] = []
    for variable in (*PDE_VARIABLES, "physics_all"):
        matrix_rmses = []
        pure_rmses = []
        constant_rmses = []
        for lead in range(1, horizon_seen + 1):
            item = stats[(variable, lead)]
            count = int(item["count"])
            matrix_rmse = torch.sqrt(item["matrix_sum_mse"] / count).mean()
            pure_rmse = torch.sqrt(item["pure_sum_mse"] / count).mean()
            constant_rmse = torch.sqrt(item["constant_sum_mse"] / count).mean()
            matrix_value = float(matrix_rmse.cpu())
            pure_value = float(pure_rmse.cpu())
            constant_value = float(constant_rmse.cpu())
            improvement_vs_constant = (
                100.0 * (constant_value - matrix_value) / constant_value
                if constant_value != 0.0
                else float("nan")
            )
            improvement_vs_pure = (
                100.0 * (pure_value - matrix_value) / pure_value
                if pure_value != 0.0
                else float("nan")
            )
            rows.append(
                {
                    "variable": variable,
                    "lead": lead,
                    "matrix_wrmse": matrix_value,
                    "pure_wrmse": pure_value,
                    "constant_wrmse": constant_value,
                    "matrix_vs_constant_improvement_pct": improvement_vs_constant,
                    "matrix_vs_pure_improvement_pct": improvement_vs_pure,
                }
            )
            matrix_rmses.append(matrix_value)
            pure_rmses.append(pure_value)
            constant_rmses.append(constant_value)

        matrix_mean = sum(matrix_rmses) / len(matrix_rmses)
        pure_mean = sum(pure_rmses) / len(pure_rmses)
        constant_mean = sum(constant_rmses) / len(constant_rmses)
        mean_improvement_vs_constant = (
            100.0 * (constant_mean - matrix_mean) / constant_mean
            if constant_mean != 0.0
            else float("nan")
        )
        mean_improvement_vs_pure = (
            100.0 * (pure_mean - matrix_mean) / pure_mean
            if pure_mean != 0.0
            else float("nan")
        )
        rows.append(
            {
                "variable": variable,
                "lead": "mean",
                "matrix_wrmse": matrix_mean,
                "pure_wrmse": pure_mean,
                "constant_wrmse": constant_mean,
                "matrix_vs_constant_improvement_pct": mean_improvement_vs_constant,
                "matrix_vs_pure_improvement_pct": mean_improvement_vs_pure,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not append or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def load_yaml_config(path: str) -> dict[str, Any]:
    from tools.evaluate_pure_pde_operator import load_config

    return load_config(path)


def best_metric(rows: list[dict[str, float | int | str]]) -> tuple[float, float, float, float, float]:
    for row in rows:
        if row["variable"] == "physics_all" and row["lead"] == "mean":
            return (
                float(row["matrix_wrmse"]),
                float(row["pure_wrmse"]),
                float(row["constant_wrmse"]),
                float(row["matrix_vs_constant_improvement_pct"]),
                float(row["matrix_vs_pure_improvement_pct"]),
            )
    raise RuntimeError("Missing physics_all mean row")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--identity-lambda", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--training-mode",
        choices=("autoregressive", "tendency"),
        default="autoregressive",
        help=(
            "autoregressive rolls the corrected PDE state forward through target leads; "
            "tendency trains/evaluates teacher-forced one-hour RHS corrections without "
            "feeding predictions back."
        ),
    )
    parser.add_argument("--matrix-mode", choices=("diagonal", "per_level", "full"), default="per_level")
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--q-scale", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--substeps-per-frame", type=int, default=12)
    parser.add_argument("--block-dt", type=float, default=300.0)
    parser.add_argument("--stencil", choices=("fd4", "weno5"), default="fd4")
    parser.add_argument(
        "--coriolis",
        choices=("constant", "beta_plane", "spherical"),
        default="spherical",
    )
    parser.add_argument("--time-scheme", choices=("euler",), default="euler")
    parser.add_argument("--boundary-x", choices=("periodic", "reflect", "replicate"), default="periodic")
    parser.add_argument("--boundary-y", choices=("periodic", "reflect", "replicate"), default="replicate")
    parser.add_argument("--boundary-z", choices=("reflect", "replicate"), default="replicate")
    parser.add_argument(
        "--lat-range-deg",
        nargs=2,
        type=float,
        default=[24.0, 56.0],
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--humidity-mode",
        choices=("relative_to_specific", "relative_as_q"),
        default="relative_to_specific",
    )
    parser.add_argument("--use-universal-R", action="store_true")
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints/pde_matrix_operator/default")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[pde-matrix] CUDA unavailable, falling back to CPU.", flush=True)
        args.device = "cpu"
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_yaml_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(config, "train", args, shuffle=True)
    val_loader = make_loader(config, "val", args, shuffle=False)
    train_dataset = train_loader.dataset
    returns_normalized = getattr(train_dataset, "returns_normalized", True)
    normalize = make_normalizer(train_dataset, device)

    first_x, _ = _to_physical_batch(_to_device(next(iter(train_loader)), device), normalize, returns_normalized)
    kernel, _ = build_kernel(args, height=first_x.shape[-2], width=first_x.shape[-1], device=device)
    scales = make_scales(train_dataset, args.q_scale, device)
    corrector = TendencyMatrixCorrector(scales=scales, mode=args.matrix_mode, max_delta=args.max_delta).to(device)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[pde-matrix] mode={args.matrix_mode}, params={sum(p.numel() for p in corrector.parameters())}, "
        f"max_delta={args.max_delta}, identity_lambda={args.identity_lambda}, "
        f"training_mode={args.training_mode}",
        flush=True,
    )
    print(
        f"[pde-matrix] dataset_returns_normalized={returns_normalized}; "
        f"kernel={args.stencil}/{args.coriolis}/{args.time_scheme}; "
        f"block_dt={args.block_dt}; substeps_per_frame={args.substeps_per_frame}",
        flush=True,
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            train_loader,
            kernel,
            corrector,
            normalize,
            optimizer,
            args,
            returns_normalized,
            scales,
        )
        val_rows = evaluate(
            val_loader,
            kernel,
            corrector,
            normalize,
            args,
            returns_normalized,
            max_batches=args.val_max_batches,
        )
        (
            matrix_wrmse,
            pure_wrmse,
            constant_wrmse,
            improvement_vs_constant,
            improvement_vs_pure,
        ) = best_metric(val_rows)
        is_best = matrix_wrmse < best_val
        if is_best:
            best_val = matrix_wrmse
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "corrector": corrector.state_dict(),
                    "args": vars(args),
                    "scales": scales.detach().cpu(),
                    "best_val_matrix_wrmse": best_val,
                },
                out_dir / "best.pt",
            )

        log_row = {
            "epoch": epoch,
            **train_metrics,
            "val_matrix_wrmse": matrix_wrmse,
            "val_pure_wrmse": pure_wrmse,
            "val_constant_wrmse": constant_wrmse,
            "val_matrix_vs_constant_improvement_pct": improvement_vs_constant,
            "val_matrix_vs_pure_improvement_pct": improvement_vs_pure,
            "best_epoch": best_epoch,
            "is_best": int(is_best),
        }
        write_rows(out_dir / "train_log.csv", [log_row], append=True)
        write_rows(out_dir / f"val_metrics_epoch_{epoch:04d}.csv", val_rows)
        print(
            f"[epoch {epoch:04d}] train_loss={train_metrics['train_loss']:.6g} "
            f"val_matrix_wrmse={matrix_wrmse:.6g} "
            f"pure={pure_wrmse:.6g} constant={constant_wrmse:.6g} "
            f"vs_pure={improvement_vs_pure:.3f}% vs_constant={improvement_vs_constant:.3f}% "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    torch.save(
        {
            "epoch": args.epochs,
            "corrector": corrector.state_dict(),
            "args": vars(args),
            "scales": scales.detach().cpu(),
            "best_epoch": best_epoch,
            "best_val_matrix_wrmse": best_val,
        },
        out_dir / "last.pt",
    )
    print(f"[pde-matrix] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
