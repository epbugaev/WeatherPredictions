"""``AutoregressiveStep`` — feeds the model's last output back in for a second pass.

Replaces ``LitModels.multiout_double``. Training is the same timestep
selection as ``TimestepSelectStep`` (hours ``[0, 2, 5]``); the validation step
runs the model twice — once on ``x`` and once on ``y_hat[:, -1, ...]`` — and
concatenates the two outputs to cover the full target horizon.
"""

from __future__ import annotations

import torch
from torch import nn

from training_strategies._index_maps import MULTIOUT_INDEX_MAP, VIS_VARS
from training_strategies._log_figures import log_prediction_maps
from training_strategies.base import StepContext, StepStrategy
from utils.registry import register_strategy

_TRAIN_HOURS: tuple[int, ...] = (0, 2, 5)


@register_strategy("multiout_double")
class AutoregressiveStep(StepStrategy):
    """Train on selected hours; validate by chaining two forwards autoregressively."""

    def __init__(self, log_figures_once: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_figures_once = log_figures_once
        self._figures_logged = False

    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = model(x)[:, _TRAIN_HOURS, ...]
        y_sel = y[:, _TRAIN_HOURS, ...]
        loss = self.loss(y_hat, y_sel)
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=loss.device)
        return {"loss": loss, "lr": lr}

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = model(x)
        y_hat_second = model(y_hat[:, -1, ...])
        y_hat_full = torch.concat([y_hat, y_hat_second], dim=1)
        val_loss = self.loss(y_hat_full, y)

        metrics = self._build_val_metrics(
            ctx,
            val_loss,
            pred_first=y_hat_full[:, 0],
            target_first=y[:, 0],
            pred_last=y_hat_full[:, -1],
            target_last=y[:, -1],
            index_map=MULTIOUT_INDEX_MAP,
        )

        if ctx.is_main_process and (not self.log_figures_once or not self._figures_logged):
            log_prediction_maps(
                experiment=ctx.experiment,
                pred=y_hat_full[:, -1, ...],
                truth=y[:, -1],
                index_map=MULTIOUT_INDEX_MAP,
                vis_vars=VIS_VARS,
                step=ctx.global_step,
            )
            self._figures_logged = True

        return metrics
