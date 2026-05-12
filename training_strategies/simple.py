"""``SimpleStep`` — the basic ``y_hat = model(x); loss = L(y_hat, y)`` step.

Replaces ``LitModels.mutiout_f`` (and the bare ``LitModels.basemodel`` shape).
The training step computes a single forward pass and a single loss; the
validation step additionally reports per-variable RMSE for the first and last
predicted timesteps.
"""

from __future__ import annotations

import torch
from torch import nn

from training_strategies._index_maps import BASEMODEL_INDEX_MAP
from training_strategies.base import StepContext, StepStrategy
from utils.registry import register_strategy


@register_strategy("mutiout_f")
class SimpleStep(StepStrategy):
    """One forward pass + per-variable RMSE on the first/last timesteps."""

    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        """Forward, loss, return.

        Args:
            model: Forward-able model.
            batch: ``(x, y)`` where ``y`` has shape ``(B, T, C, H, W)``.
            ctx: Per-step context.

        Returns:
            ``{"loss": ..., "lr": ...}``; the trainer logs both.
        """
        x, y = batch
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=loss.device)
        return {"loss": loss, "lr": lr}

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        """Forward, loss, per-variable RMSE on first and last timestep.

        Args:
            model: Forward-able model.
            batch: ``(x, y)`` of shape ``(B, T, C, H, W)``.
            ctx: Per-step context.

        Returns:
            ``{"val_loss": ..., "RMSE_<var>_first": ..., "RMSE_<var>_last": ...}``.
        """
        x, y = batch
        y_hat = model(x)
        val_loss = self.loss(y_hat, y)
        rmse_first = ctx.metrics.WRMSE(y_hat[:, 0], y[:, 0])
        rmse_last = ctx.metrics.WRMSE(y_hat[:, -1], y[:, -1])

        metrics: dict[str, torch.Tensor] = {"val_loss": val_loss}
        for var_name, idx in BASEMODEL_INDEX_MAP.items():
            metrics[f"RMSE_{var_name}_first"] = torch.as_tensor(
                rmse_first[idx], device=val_loss.device
            )
            metrics[f"RMSE_{var_name}_last"] = torch.as_tensor(
                rmse_last[idx], device=val_loss.device
            )
        return metrics
