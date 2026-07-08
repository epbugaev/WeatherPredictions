"""Per-channel z-score normalization for ERA5 tensors.

Stores ``mean``/``std`` as ``register_buffer`` so they:
  * round-trip through ``state_dict`` (one source of truth),
  * follow ``module.to(device)`` and DDP replication automatically,
  * stay fp32 under ``torch.amp.autocast`` — autocast does not cast
    buffers; the arithmetic in ``forward`` is promoted to the input
    dtype and autocast returns to fp16/bf16 as appropriate.
"""

from __future__ import annotations

import torch
from torch import nn


class WeatherNormalize(nn.Module):
    """``forward`` = ``(x - mean) / std`` (normalize); ``denormalize`` — обратное.

    ``mean``/``std`` are stored with shape ``(C, 1, 1)`` so they broadcast
    over any number of leading dims, e.g. ``(B, T, C, H, W)`` or
    ``(B, S, T, C, H, W)``.

    Note:
        ``WeatherNormalize.denormalize`` performs the inverse operation,
        ``raw * std + mean``, when physical-unit tensors are needed.

    Args:
        mean: 1-D tensor of length ``C``.
        std:  1-D tensor of length ``C``.

    Raises:
        ValueError: if ``mean`` or ``std`` is not 1-D, or shapes mismatch.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(
                f"mean/std must be 1-D; got mean.ndim={mean.ndim}, std.ndim={std.ndim}"
            )
        if mean.shape != std.shape:
            raise ValueError(
                f"mean and std must have matching shapes; got {mean.shape} vs {std.shape}"
            )
        mean_buf = mean.detach().clone().float().view(-1, 1, 1).contiguous()
        std_buf = std.detach().clone().float().view(-1, 1, 1).contiguous()
        self.register_buffer("mean", mean_buf)
        self.register_buffer("std", std_buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score; preserves ``x`` shape and (under autocast) dtype."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of ``forward`` — back to physical units."""
        return x * self.std + self.mean
