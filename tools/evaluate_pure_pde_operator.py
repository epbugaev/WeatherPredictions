#!/usr/bin/env python3
"""Evaluate equations-only PurePDEKernel against persistence.

This is the narrowest physics diagnostic for the v4 experiments. It does not
build IAM4VP, WeatherGFT, HybridBlock, Conv2d adapters, residual heads, or load
any checkpoint. It only:

1. reads WeatherBench frames from the configured dataset,
2. takes the last input frame as the initial physical state,
3. converts ERA5 relative humidity ``r`` to a specific-humidity proxy ``q``,
4. rolls ``utils.physics.PurePDEKernel`` forward, and
5. compares the pure-PDE forecast with the constant baseline.

Reported variables are exactly the prognostic upper-air fields used by the PDE:
``z, t, q, u, v``. Surface channels are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import Data  # noqa: E402,F401 - registers datasets
from utils.normalize import WeatherNormalize  # noqa: E402
from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402
from utils.registry import get_dataset  # noqa: E402


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

CHANNEL_RANGES: dict[str, tuple[int, int]] = {
    "z": (4, 17),
    "t": (17, 30),
    "r": (30, 43),
    "u": (43, 56),
    "v": (56, 69),
}

PDE_VARIABLES = ("z", "t", "q", "u", "v")


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return _expand_env_value(yaml.safe_load(f))


def build_dataset(data_cfg: dict[str, Any], split: str):
    version = data_cfg.get("dataset_version", "v3")
    split_cfg = data_cfg.get(split, {})

    params: dict[str, Any] = {
        "start_time": split_cfg["start_time"],
        "end_time": split_cfg["end_time"],
        "include_target": split_cfg.get("include_target", data_cfg.get("include_target", False)),
        "lead_time": split_cfg.get("lead_time", data_cfg.get("lead_time", 1)),
        "interval": split_cfg.get("interval", data_cfg.get("interval", 1)),
        "muti_target_steps": split_cfg.get(
            "muti_target_steps",
            data_cfg.get("muti_target_steps", 1),
        ),
        "start_time_x": data_cfg.get("start_time_x", 0),
        "end_time_x": data_cfg.get("end_time_x", 1),
        "start_time_y": data_cfg.get("start_time_y", 0),
        "end_time_y": data_cfg.get("end_time_y", 1),
    }

    for key in ("sample_stride", "frame_interval"):
        value = split_cfg.get(key, data_cfg.get(key))
        if value is not None:
            params[key] = value

    cut = split_cfg.get("cut", data_cfg.get("cut"))
    if cut is not None:
        params["cut"] = cut

    for key in (
        "data_folder",
        "input_folder",
        "mean_std_path",
        "memmap_path",
        "memmap_meta_path",
    ):
        env_key = key.upper() + "_OVERRIDE"
        env_value = os.environ.get(env_key)
        if env_value:
            params[key] = env_value
            continue
        if key == "data_folder":
            env_value = os.environ.get("WEATHERBENCH_NPY_ROOT")
        elif key == "input_folder":
            env_value = os.environ.get("WEATHERBENCH_INPUT_ROOT")
        elif key == "mean_std_path":
            env_value = os.environ.get("WEATHERBENCH_MEAN_STD_PATH")
        if env_value:
            params[key] = env_value
            continue
        value = data_cfg.get(key)
        if value is not None:
            params[key] = value

    return get_dataset(version)(**params)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        return type(batch)(_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    return batch


def _make_normalizer(dataset, device: torch.device) -> WeatherNormalize:
    return WeatherNormalize(
        mean=torch.as_tensor(dataset.the_mean, dtype=torch.float32),
        std=torch.as_tensor(dataset.the_std, dtype=torch.float32),
    ).to(device)


def _to_physical_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    normalize: WeatherNormalize,
    returns_normalized: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if returns_normalized:
        return normalize.denormalize(x), normalize.denormalize(y)
    return x, y


def split_channels(frame: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a physical ``(B, 69, H, W)`` frame into PDE channels."""
    if frame.dim() != 4 or frame.shape[1] != 69:
        raise ValueError(f"Expected frame shape (B,69,H,W), got {tuple(frame.shape)}")
    return {name: frame[:, start:end] for name, (start, end) in CHANNEL_RANGES.items()}


def relative_to_specific_humidity(
    kernel: PurePDEKernel,
    r_percent: torch.Tensor,
    temperature: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "relative_as_q":
        return r_percent
    pressure = kernel.grid.pressure.to(device=temperature.device, dtype=temperature.dtype)
    q_s = kernel._get_qs(pressure.expand_as(temperature), temperature)
    return (r_percent / 100.0).clamp(min=0.0, max=2.0) * q_s


def physical_state_from_frame(
    frame: torch.Tensor,
    kernel: PurePDEKernel,
    humidity_mode: str,
) -> dict[str, torch.Tensor]:
    parts = split_channels(frame)
    q = relative_to_specific_humidity(kernel, parts["r"], parts["t"], humidity_mode)
    return {
        "z": parts["z"],
        "t": parts["t"],
        "q": q,
        "u": parts["u"],
        "v": parts["v"],
    }


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state.items()}


def pde_step(
    kernel: PurePDEKernel,
    state: dict[str, torch.Tensor],
    substeps: int,
) -> dict[str, torch.Tensor]:
    cur = state
    for _ in range(substeps):
        step = kernel.step(cur["u"], cur["v"], cur["t"], cur["q"], cur["z"])
        cur = {key: step[key] for key in PDE_VARIABLES}
    return cur


def weighted_mse_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted MSE per sample and pressure level.

    Args:
        pred, target: ``(B, P, H, W)``.
        lat_weights: normalized ``(H,)`` weights whose sum is ``H``.

    Returns:
        Tensor ``(B, P)``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    _, _, height, width = pred.shape
    weights = lat_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    weight_sum = lat_weights.sum().to(device=pred.device, dtype=pred.dtype) * width
    return (weights * (pred - target).square()).sum(dim=(2, 3)) / weight_sum


def _add_metric(
    stats: dict[tuple[str, int], dict[str, torch.Tensor | int]],
    variable: str,
    lead: int,
    pde_pred: torch.Tensor,
    const_pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
) -> None:
    pde_mse = weighted_mse_per_channel(pde_pred, target, lat_weights)
    const_mse = weighted_mse_per_channel(const_pred, target, lat_weights)
    key = (variable, lead)
    if key not in stats:
        stats[key] = {
            "pde_sum_mse": torch.zeros(pde_mse.shape[1], device=pde_mse.device),
            "const_sum_mse": torch.zeros(const_mse.shape[1], device=const_mse.device),
            "count": 0,
        }
    stats[key]["pde_sum_mse"] = stats[key]["pde_sum_mse"] + pde_mse.sum(dim=0)
    stats[key]["const_sum_mse"] = stats[key]["const_sum_mse"] + const_mse.sum(dim=0)
    stats[key]["count"] = int(stats[key]["count"]) + pde_mse.shape[0]


def build_kernel(args: argparse.Namespace, height: int, width: int, device: torch.device):
    grid = Grid(GridConfig(H=height, W=width, lat_range_deg=tuple(args.lat_range_deg))).to(device)
    kernel = PurePDEKernel(
        grid,
        stencil=args.stencil,
        coriolis=args.coriolis,
        block_dt=args.block_dt,
        time_scheme=args.time_scheme,
        boundary_x=args.boundary_x,
        boundary_y=args.boundary_y,
        boundary_z=args.boundary_z,
        use_universal_R=args.use_universal_R,
    ).to(device)
    lat_weights = height * torch.cos(grid.latitudes) / torch.cos(grid.latitudes).sum()
    return kernel, lat_weights


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    config = load_config(args.config)
    device = torch.device(args.device)
    dataset = build_dataset(config["data"], args.split)
    returns_normalized = getattr(dataset, "returns_normalized", True)
    normalize = _make_normalizer(dataset, device)
    batch_size = args.batch_size or config["data"].get(args.split, {}).get("batch_size", 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    stats: dict[tuple[str, int], dict[str, torch.Tensor | int]] = {}
    kernel: PurePDEKernel | None = None
    lat_weights: torch.Tensor | None = None
    horizon_seen: int | None = None
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        x, y = _to_physical_batch(_to_device(batch, device), normalize, returns_normalized)
        if x.dim() != 5 or y.dim() != 5:
            raise ValueError(f"Expected x/y shapes (B,T,C,H,W), got {tuple(x.shape)} and {tuple(y.shape)}")
        if x.shape[2] != 69 or y.shape[2] != 69:
            raise ValueError(f"Expected 69 channels, got x={x.shape[2]}, y={y.shape[2]}")

        if kernel is None or lat_weights is None:
            kernel, lat_weights = build_kernel(args, height=x.shape[-2], width=x.shape[-1], device=device)
            print(
                "[pure-pde] channel layout: surface=0:4, z=4:17, t=17:30, "
                "r=30:43, u=43:56, v=56:69",
                flush=True,
            )
            print(
                f"[pure-pde] dataset_returns_normalized={returns_normalized}; "
                f"humidity_mode={args.humidity_mode}; grid={x.shape[-2]}x{x.shape[-1]}; "
                f"lat_range_deg={tuple(args.lat_range_deg)}",
                flush=True,
            )
            print(
                f"[pure-pde] kernel: stencil={args.stencil}, coriolis={args.coriolis}, "
                f"time_scheme={args.time_scheme}, block_dt={args.block_dt}, "
                f"substeps_per_frame={args.substeps_per_frame}",
                flush=True,
            )

        horizon = min(args.horizon or y.shape[1], y.shape[1])
        if horizon_seen is None:
            horizon_seen = horizon
        elif horizon_seen != horizon:
            raise ValueError("Variable horizon across batches is not supported")

        initial_state = physical_state_from_frame(x[:, -1], kernel, args.humidity_mode)
        pde_state = clone_state(initial_state)
        constant_state = clone_state(initial_state)

        for lead_idx in range(horizon):
            pde_state = pde_step(kernel, pde_state, args.substeps_per_frame)
            target_state = physical_state_from_frame(y[:, lead_idx], kernel, args.humidity_mode)

            for variable in PDE_VARIABLES:
                _add_metric(
                    stats,
                    variable,
                    lead_idx + 1,
                    pde_state[variable],
                    constant_state[variable],
                    target_state[variable],
                    lat_weights,
                )

            pde_all = torch.cat([pde_state[variable] for variable in PDE_VARIABLES], dim=1)
            constant_all = torch.cat([constant_state[variable] for variable in PDE_VARIABLES], dim=1)
            target_all = torch.cat([target_state[variable] for variable in PDE_VARIABLES], dim=1)
            _add_metric(
                stats,
                "physics_all",
                lead_idx + 1,
                pde_all,
                constant_all,
                target_all,
                lat_weights,
            )

        num_batches += 1
        if args.log_every and num_batches % args.log_every == 0:
            print(f"[pure-pde] processed {num_batches} batches", flush=True)

    if num_batches == 0 or horizon_seen is None:
        raise RuntimeError("No batches evaluated")

    rows: list[dict[str, float | int | str]] = []
    for variable in (*PDE_VARIABLES, "physics_all"):
        pde_rmses = []
        const_rmses = []
        for lead in range(1, horizon_seen + 1):
            item = stats[(variable, lead)]
            count = int(item["count"])
            pde_rmse_by_level = torch.sqrt(item["pde_sum_mse"] / count)
            const_rmse_by_level = torch.sqrt(item["const_sum_mse"] / count)
            pde_rmse = float(pde_rmse_by_level.mean().item())
            const_rmse = float(const_rmse_by_level.mean().item())
            improvement = 100.0 * (const_rmse - pde_rmse) / const_rmse if const_rmse != 0.0 else float("nan")
            rows.append(
                {
                    "variable": variable,
                    "lead": lead,
                    "pde_wrmse": pde_rmse,
                    "constant_wrmse": const_rmse,
                    "pde_improvement_pct": improvement,
                }
            )
            pde_rmses.append(pde_rmse)
            const_rmses.append(const_rmse)

        pde_mean = sum(pde_rmses) / len(pde_rmses)
        const_mean = sum(const_rmses) / len(const_rmses)
        improvement_mean = 100.0 * (const_mean - pde_mean) / const_mean if const_mean != 0.0 else float("nan")
        rows.append(
            {
                "variable": variable,
                "lead": "mean",
                "pde_wrmse": pde_mean,
                "constant_wrmse": const_mean,
                "pde_improvement_pct": improvement_mean,
            }
        )
    return rows


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = ["variable", "lead", "pde_wrmse", "constant_wrmse", "pde_improvement_pct"]
    print("\t".join(headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        print("\t".join(values))


def write_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    headers = ["variable", "lead", "pde_wrmse", "constant_wrmse", "pde_improvement_pct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config used only for data loading.")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, help="Number of target frames to evaluate.")
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=12,
        help="PurePDEKernel substeps per target frame. With block_dt=300 this is 1 hour.",
    )
    parser.add_argument("--block-dt", type=float, default=300.0)
    parser.add_argument("--stencil", choices=("fd4", "weno5"), default="fd4")
    parser.add_argument(
        "--coriolis",
        choices=("constant", "beta_plane", "spherical"),
        default="spherical",
    )
    parser.add_argument(
        "--time-scheme",
        choices=("euler", "rk4", "ssp_rk3", "semi_implicit"),
        default="euler",
    )
    parser.add_argument(
        "--boundary-x",
        choices=("periodic", "reflect", "replicate"),
        default="periodic",
    )
    parser.add_argument(
        "--boundary-y",
        choices=("periodic", "reflect", "replicate"),
        default="replicate",
    )
    parser.add_argument("--boundary-z", choices=("reflect", "replicate"), default="replicate")
    parser.add_argument(
        "--lat-range-deg",
        nargs=2,
        type=float,
        default=[24.0, 56.0],
        metavar=("LOW", "HIGH"),
        help="Latitude range represented by the crop. For the USA v4 crop use 24 56.",
    )
    parser.add_argument(
        "--humidity-mode",
        choices=("relative_to_specific", "relative_as_q"),
        default="relative_to_specific",
        help="Default converts ERA5 relative humidity r(%) to a q proxy before PDE.",
    )
    parser.add_argument(
        "--use-universal-R",
        action="store_true",
        help="Opt into the old WeatherGFT R=8.314 hydrostatic behavior.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[pure-pde] CUDA unavailable, falling back to CPU.", flush=True)
        args.device = "cpu"
    return args


def main() -> None:
    args = parse_args()
    rows = evaluate(args)
    print_table(rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"[pure-pde] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
