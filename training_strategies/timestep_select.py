"""``TimestepSelectStep`` — train on a subset of timesteps ``[0, 2, 5]``.

Replaces ``LitModels.mutiout``. The training step picks timesteps ``[0, 2, 5]``
from both prediction and target before computing the loss; the validation step
uses the model's ``out_layer`` indices for slicing ``y`` and logs prediction
maps for ``vis_vars``.
"""

from __future__ import annotations

import torch
from torch import nn

from training_strategies._index_maps import MULTIOUT_INDEX_MAP, VIS_VARS
from training_strategies._log_figures import log_prediction_maps
from training_strategies.base import StepContext, StepStrategy
from utils.registry import register_strategy

_TRAIN_HOURS: tuple[int, ...] = (0, 2, 5)


def _model_out_indices(model: nn.Module) -> list[int] | None:
    """Look up ``out_layer`` on the raw model (un-wrap DDP if needed)."""
    raw = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if hasattr(raw, "out_layer"):
        return raw.out_layer
    return None


@register_strategy("mutiout")
class TimestepSelectStep(StepStrategy):
    """Loss only on selected output timesteps + figure logging on validation."""

    def __init__(self, log_figures_once: bool = True, **kwargs) -> None:
        """Configure timestep selection and figure-logging cadence.

        Args:
            log_figures_once: If True, prediction maps are logged on the first
                validation epoch only (mirrors the legacy ``self.trained``
                flag in LitModels). Set False to log every validation.
        """
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
        y_sel = y[:, _TRAIN_HOURS, ...]
        y_hat = model(x)[:, _TRAIN_HOURS, ...]
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
        out_idx = _model_out_indices(model)
        if out_idx is None:
            raise AttributeError(
                "TimestepSelectStep expects model.out_layer to define output indices"
            )
        y_sel = y[:, out_idx]
        y_hat = model(x)
        val_loss = self.loss(y_hat, y_sel)

        metrics = self._build_val_metrics(
            ctx,
            val_loss,
            pred_first=y_hat[:, 0],
            target_first=y_sel[:, 0],
            pred_last=y_hat[:, -1],
            target_last=y_sel[:, -1],
            index_map=MULTIOUT_INDEX_MAP,
            prefix="f ",
        )

        if ctx.is_main_process and (not self.log_figures_once or not self._figures_logged):
            log_prediction_maps(
                experiment=ctx.experiment,
                pred=y_hat[:, -1, ...],
                truth=y_sel[:, -1],
                index_map=MULTIOUT_INDEX_MAP,
                vis_vars=VIS_VARS,
                step=ctx.global_step,
            )
            self._figures_logged = True

        return metrics
