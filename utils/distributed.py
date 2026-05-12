"""Pure-PyTorch distributed helpers for DDP training.

``setup_distributed`` reads the ``RANK``, ``LOCAL_RANK`` and ``WORLD_SIZE``
environment variables that ``torchrun`` sets, calls
``torch.distributed.init_process_group`` and binds the current process to
the right GPU.

Single-process (non-distributed) runs are detected when ``WORLD_SIZE`` is
absent or equals 1 — in that case the helpers degrade to no-ops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    """Distributed runtime info, available even when DDP is not active.

    Attributes:
        rank: Global rank across all nodes (0 in single-process mode).
        local_rank: Rank within the current node (0 in single-process mode).
        world_size: Total number of processes (1 in single-process mode).
        is_distributed: Whether ``torch.distributed`` is initialised.
        device: ``torch.device`` bound to this process.
    """

    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    device: torch.device


def setup_distributed(backend: str = "nccl") -> DistInfo:
    """Initialise the process group from torchrun environment variables.

    Reads ``RANK``, ``LOCAL_RANK`` and ``WORLD_SIZE`` (set by ``torchrun``).
    If ``WORLD_SIZE`` is missing or equals 1, returns a single-process
    ``DistInfo`` without touching ``torch.distributed``.

    Args:
        backend: Process-group backend; ``nccl`` for GPU, ``gloo`` for CPU.

    Returns:
        ``DistInfo`` describing the current process.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return DistInfo(rank=0, local_rank=0, world_size=1, is_distributed=False, device=device)

    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=True,
        device=device,
    )


def cleanup_distributed() -> None:
    """Tear down the process group if it was initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(info: DistInfo | None = None) -> bool:
    """True only on global rank 0 (or in single-process mode)."""
    if info is not None:
        return info.rank == 0
    return int(os.environ.get("RANK", "0")) == 0


def barrier(info: DistInfo) -> None:
    """Synchronise all ranks; no-op in single-process mode."""
    if info.is_distributed and dist.is_initialized():
        dist.barrier()


def reduce_dict(
    metrics: dict[str, torch.Tensor],
    info: DistInfo,
    average: bool = True,
) -> dict[str, torch.Tensor]:
    """All-reduce a dict of scalar tensors across ranks.

    Args:
        metrics: Mapping of metric name to 0-dim ``torch.Tensor``. Must have
            identical keys on every rank.
        info: Distributed info from ``setup_distributed``.
        average: If True, divide the reduced sum by ``world_size``.

    Returns:
        New dict with reduced tensors. Returns ``metrics`` unchanged in
        single-process mode.
    """
    if not info.is_distributed:
        return metrics

    keys = sorted(metrics.keys())
    values = torch.stack([metrics[k].detach().float() for k in keys])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values = values / info.world_size
    return {k: values[i] for i, k in enumerate(keys)}
