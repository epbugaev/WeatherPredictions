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
            ``{"loss": ..., "lr": ...}``; the trainer logs both. For PI-models
            (PI-SimVPv2, exp21) the physics L1 penalty is added to the task loss
            and both terms are reported separately — ``forecast_loss``,
            ``physics_residual_aux_loss`` — alongside the physics diagnostics,
            so a physics branch degrading to its sanitize fallback stays visible.
            Models without a physics branch keep the plain two-key dict.
        """
        x, y = batch
        y_hat = model(x)
        forecast_loss = self.loss(y_hat, y)
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=forecast_loss.device)
        if not hasattr(self._inner_model(model), "physics_residual_aux_loss"):
            return {"loss": forecast_loss, "lr": lr}

        aux_loss = self._physics_residual_aux_loss(model, forecast_loss.device)
        metrics = {
            "loss": forecast_loss + aux_loss,
            "forecast_loss": forecast_loss.detach(),
            "physics_residual_aux_loss": aux_loss.detach(),
            "lr": lr,
        }
        metrics.update(self._physics_residual_diagnostics(model))
        return metrics

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
        return self._build_val_metrics(
            ctx,
            val_loss,
            pred_first=y_hat[:, 0],
            target_first=y[:, 0],
            pred_last=y_hat[:, -1],
            target_last=y[:, -1],
            index_map=BASEMODEL_INDEX_MAP,
            pred_full=y_hat,
            target_full=y,
        )
