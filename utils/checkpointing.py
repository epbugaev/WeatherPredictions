"""Checkpoint save/load + converter for legacy Lightning ``.ckpt`` files.

The native format is a flat ``torch.save`` dict::

    {
        "model": state_dict,            # weights only, no ``model.`` prefix
        "optimizer": optimizer_state,   # optional
        "scheduler": scheduler_state,   # optional
        "scaler": amp_scaler_state,     # optional
        "epoch": int,
        "global_step": int,
        "metric": float,                # value of the monitored metric
        "config": dict,                 # the YAML config the run was started with
    }

Loading is robust to missing optional keys (LBYL: ``if "optimizer" in ckpt``)
so the same loader works for ``best.pt``, ``last.pt`` and for inference where
only the model weights are needed.

A standalone CLI converts Lightning ``.ckpt`` files (where weights live under
``ckpt["state_dict"]`` with a ``model.`` prefix from the LightningModule
wrapper) into the native format.

CLI::

    python -m utils.checkpointing convert <src.ckpt> <dst.pt>
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointMeta:
    """Metadata returned by ``load_checkpoint`` after restoring states."""

    epoch: int
    global_step: int
    metric: float
    config: dict[str, Any]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    metric: float,
    config: dict[str, Any],
) -> None:
    """Save a checkpoint in the native flat format.

    Unwraps ``DistributedDataParallel`` automatically so the saved state-dict
    has no ``module.`` prefix. The directory is created if it does not exist.
    Caller is responsible for invoking this only from the main rank.

    Args:
        path: Output file path (``.pt`` recommended).
        model: Model to save; may be wrapped in DDP.
        optimizer: Optional optimiser state to include.
        scheduler: Optional LR-scheduler state to include.
        scaler: Optional AMP grad scaler state to include.
        epoch: Epoch number that just finished.
        global_step: Total number of optimiser steps taken so far.
        metric: Value of the monitored metric (for filename templating).
        config: Top-level YAML config dict for traceability.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    payload: dict[str, Any] = {
        "model": raw_model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metric": metric,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointMeta:
    """Load a native checkpoint into ``model`` (and optionally optimiser/scheduler).

    Args:
        path: Checkpoint file path.
        model: Model whose ``state_dict`` should be restored.
        optimizer: Optional optimiser to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional AMP grad scaler to restore.
        map_location: ``torch.load`` map_location argument.
        strict: ``load_state_dict`` strictness for the model.

    Returns:
        ``CheckpointMeta`` populated from the checkpoint's metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    raw_model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return CheckpointMeta(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        metric=float(ckpt.get("metric", float("inf"))),
        config=dict(ckpt.get("config", {})),
    )


def convert_lightning_checkpoint(
    src: str,
    dst: str,
    keep_optimizer: bool = False,
    map_location: str | torch.device = "cpu",
) -> None:
    """Convert a Lightning ``.ckpt`` to the native flat format.

    Lightning stores weights under ``ckpt["state_dict"]`` with a ``model.``
    prefix (because the LightningModule wraps the actual ``nn.Module`` in
    ``self.model``). This function strips the prefix and writes a native
    checkpoint that ``load_checkpoint`` can read directly.

    Args:
        src: Path to the legacy ``.ckpt`` file.
        dst: Path where the native checkpoint will be written.
        keep_optimizer: If True and the Lightning checkpoint contains
            optimizer states, copy them into the new file.
        map_location: ``torch.load`` map_location argument.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    ckpt = torch.load(src, map_location=map_location)
    if "state_dict" not in ckpt:
        raise ValueError(
            f"{src} does not look like a Lightning checkpoint "
            "(no 'state_dict' key); nothing to convert."
        )

    prefix = "model."
    converted: dict[str, torch.Tensor] = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith(prefix):
            converted[key[len(prefix) :]] = value
        else:
            converted[key] = value

    payload: dict[str, Any] = {
        "model": converted,
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "metric": float("inf"),
        "config": dict(ckpt.get("hyper_parameters", {})),
    }

    if keep_optimizer and "optimizer_states" in ckpt and ckpt["optimizer_states"]:
        payload["optimizer"] = ckpt["optimizer_states"][0]
    if keep_optimizer and "lr_schedulers" in ckpt and ckpt["lr_schedulers"]:
        payload["scheduler"] = ckpt["lr_schedulers"][0]

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(payload, dst)


def _cli() -> None:
    """Entry point for ``python -m utils.checkpointing ...``."""
    parser = argparse.ArgumentParser(description="Checkpoint utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    conv = sub.add_parser("convert", help="Convert a Lightning .ckpt to the native format")
    conv.add_argument("src", help="Path to the legacy Lightning .ckpt")
    conv.add_argument("dst", help="Output path for the native .pt")
    conv.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Copy optimizer/scheduler states when present",
    )

    args = parser.parse_args()
    if args.cmd == "convert":
        convert_lightning_checkpoint(args.src, args.dst, keep_optimizer=args.keep_optimizer)


if __name__ == "__main__":
    _cli()
