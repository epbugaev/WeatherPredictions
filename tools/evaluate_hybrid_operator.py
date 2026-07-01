#!/usr/bin/env python3
"""Evaluate the PI-IAM4VP HybridBlock-only operator against persistence.

The residual-corrector experiments use HybridBlock as a physics/tendency feature
generator. This script asks a narrower question: if we take only that operator
(``_physics_prior_from_state``), how good is its direct forecast compared with
the constant baseline ``y_hat(t+k) = x_last``?

Only upper-air variables that participate in the HybridBlock physics path are
reported: z, t, r, u, v. Surface channels are intentionally ignored.
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
import Models  # noqa: E402,F401 - registers models
from utils.checkpointing import load_checkpoint  # noqa: E402
from utils.metrics import weighted_rmse_torch  # noqa: E402
from utils.normalize import WeatherNormalize  # noqa: E402
from utils.registry import get_dataset, get_model  # noqa: E402


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

PHYSICS_GROUPS: dict[str, tuple[int, int]] = {
    "z": (4, 17),
    "t": (17, 30),
    "r": (30, 43),
    "u": (43, 56),
    "v": (56, 69),
    "upper_air_all": (4, 69),
}


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


def _make_normalizer(dataset, device: torch.device) -> WeatherNormalize | None:
    if getattr(dataset, "returns_normalized", True):
        return None
    return WeatherNormalize(
        mean=torch.as_tensor(dataset.the_mean, dtype=torch.float32),
        std=torch.as_tensor(dataset.the_std, dtype=torch.float32),
    ).to(device)


def _normalize_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    normalize: WeatherNormalize | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if normalize is None:
        return x, y
    return normalize(x), normalize(y)


def _load_model(
    config: dict[str, Any],
    checkpoint: str,
    device: torch.device,
    dataset,
    normalize: WeatherNormalize | None,
    strict: bool,
):
    model_cfg = config["model"]
    model = get_model(model_cfg["type"])(**model_cfg.get("params", {}))
    load_checkpoint(checkpoint, model, map_location="cpu", strict=strict, normalize=normalize)
    model.to(device)
    model.eval()

    set_physics_normalization = getattr(model, "set_physics_normalization", None)
    if set_physics_normalization is not None:
        set_physics_normalization(dataset.data_mean_tensor, dataset.data_std_tensor)
    if not hasattr(model, "_physics_prior_from_state"):
        raise TypeError("Model does not expose _physics_prior_from_state; expected PI-IAM4VP.")
    return model


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    config = load_config(args.config)
    if args.hybrid_mode:
        config.setdefault("model", {}).setdefault("params", {})[
            "physics_residual_hybrid_mode"
        ] = args.hybrid_mode

    device = torch.device(args.device)
    dataset = build_dataset(config["data"], args.split)
    normalize = _make_normalizer(dataset, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config["data"].get(args.split, {}).get("batch_size", 1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _load_model(
        config,
        args.checkpoint,
        device,
        dataset,
        normalize,
        strict=not args.no_strict,
    )

    data_std = torch.as_tensor(dataset.data_std_tensor, dtype=torch.float32, device=device)
    sum_operator: dict[str, torch.Tensor] = {}
    sum_constant: dict[str, torch.Tensor] = {}
    num_batches = 0
    horizon_seen: int | None = None

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        x, y = _normalize_batch(_to_device(batch, device), normalize)
        if y.dim() != 5:
            raise ValueError(f"Expected y shape (B,T,C,H,W), got {tuple(y.shape)}")

        horizon = min(args.horizon or y.shape[1], y.shape[1])
        if horizon_seen is None:
            horizon_seen = horizon
        elif horizon_seen != horizon:
            raise ValueError("Variable horizon across batches is not supported")

        target = y[:, :horizon]
        prev_operator = x[:, -1]
        prev_constant = x[:, -1]
        operator_steps = []
        constant_steps = []
        for _ in range(horizon):
            pred_operator = model._physics_prior_from_state(prev_operator)
            operator_steps.append(pred_operator)
            constant_steps.append(prev_constant)
            prev_operator = pred_operator if args.autoregressive else x[:, -1]

        operator = torch.stack(operator_steps, dim=1)
        constant = torch.stack(constant_steps, dim=1)

        for group, (start, end) in PHYSICS_GROUPS.items():
            group_std = data_std[start:end].view(1, -1)
            operator_rmse = weighted_rmse_torch(
                operator[:, :, start:end],
                target[:, :, start:end],
            ) * group_std
            constant_rmse = weighted_rmse_torch(
                constant[:, :, start:end],
                target[:, :, start:end],
            ) * group_std
            sum_operator[group] = sum_operator.get(group, torch.zeros_like(operator_rmse)) + operator_rmse
            sum_constant[group] = sum_constant.get(group, torch.zeros_like(constant_rmse)) + constant_rmse

        num_batches += 1
        if args.log_every and num_batches % args.log_every == 0:
            print(f"[eval] processed {num_batches} batches", flush=True)

    if num_batches == 0 or horizon_seen is None:
        raise RuntimeError("No batches evaluated")

    rows: list[dict[str, float | int | str]] = []
    for group in PHYSICS_GROUPS:
        operator_rmse = sum_operator[group] / num_batches
        constant_rmse = sum_constant[group] / num_batches
        for lead_idx in range(horizon_seen):
            op = float(operator_rmse[lead_idx].mean().item())
            const = float(constant_rmse[lead_idx].mean().item())
            improvement = 100.0 * (const - op) / const if const != 0.0 else float("nan")
            rows.append(
                {
                    "group": group,
                    "lead": lead_idx + 1,
                    "operator_wrmse": op,
                    "constant_wrmse": const,
                    "operator_improvement_pct": improvement,
                }
            )
        op_all = float(operator_rmse.mean().item())
        const_all = float(constant_rmse.mean().item())
        improvement_all = 100.0 * (const_all - op_all) / const_all if const_all != 0.0 else float("nan")
        rows.append(
            {
                "group": group,
                "lead": "mean",
                "operator_wrmse": op_all,
                "constant_wrmse": const_all,
                "operator_improvement_pct": improvement_all,
            }
        )
    return rows


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = ["group", "lead", "operator_wrmse", "constant_wrmse", "operator_improvement_pct"]
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
    headers = ["group", "lead", "operator_wrmse", "constant_wrmse", "operator_improvement_pct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config used to build PI-IAM4VP.")
    parser.add_argument("--checkpoint", required=True, help="Native checkpoint path, e.g. best.pt.")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, help="Number of target frames to evaluate.")
    parser.add_argument(
        "--hybrid-mode",
        choices=("legacy_normalized", "stable_physical"),
        default=None,
        help="Override model.params.physics_residual_hybrid_mode from the config.",
    )
    parser.add_argument(
        "--autoregressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed operator predictions back into the next operator step.",
    )
    parser.add_argument("--no-strict", action="store_true", help="Load checkpoint with strict=False.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate(args)
    print_table(rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"[eval] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
