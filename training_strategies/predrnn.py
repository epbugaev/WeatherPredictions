"""``PredRNNStep`` — PredRNN-specific concat/permute around the forward pass.

Replaces ``LitModels.mutiout_predrnn``. PredRNN's API differs from the other
video predictors: input is ``(x, y)`` concatenated along the time axis and
permuted to channels-last; the second positional argument is the
mask-tensor used for scheduled sampling (we pass zeros, matching legacy).
"""

from __future__ import annotations

import torch
from torch import nn

from training_strategies._index_maps import MULTIOUT_INDEX_MAP, VIS_VARS
from training_strategies._log_figures import log_prediction_maps
from training_strategies.base import StepContext, StepStrategy
from utils.registry import register_strategy

def _predrnn_forward(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full ``(x, y)`` concat-permute-forward-permute pipeline.

    Returns:
        Tuple ``(inp, y_hat)`` both in ``(B, T, C, H, W)`` layout.
    """
    inp = torch.cat([x, y], dim=1)
    inp = inp.permute(0, 1, 3, 4, 2).contiguous()
    mask = torch.zeros((1, y.shape[1], 1, 1, 1), device=x.device, dtype=x.dtype)
    y_hat_perm, _ = model(inp, mask)
    y_hat = y_hat_perm.permute(0, 1, 4, 2, 3)
    inp = inp.permute(0, 1, 4, 2, 3)
    return inp, y_hat


@register_strategy("mutiout_predrnn")
class PredRNNStep(StepStrategy):
    """PredRNN forward signature plus per-variable RMSE and figure logging."""

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
        inp, y_hat = _predrnn_forward(model, x, y)
        loss = self.loss(inp[:, 1:, ...], y_hat)
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=loss.device)
        return {"loss": loss, "lr": lr}

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        _, y_hat_full = _predrnn_forward(model, x, y)
        start = x.shape[1] - 1
        y_hat = y_hat_full[:, start : start + y.shape[1], ...]
        val_loss = self.loss(y_hat, y)

        metrics = self._build_val_metrics(
            ctx,
            val_loss,
            pred_first=y_hat[:, 0],
            target_first=y[:, 0],
            pred_last=y_hat[:, -1],
            target_last=y[:, -1],
            index_map=MULTIOUT_INDEX_MAP,
            pred_full=y_hat,
            target_full=y,
        )

        if ctx.is_main_process and (not self.log_figures_once or not self._figures_logged):
            log_prediction_maps(
                experiment=ctx.experiment,
                pred=y_hat[:, -1, ...],
                truth=y[:, -1],
                index_map=MULTIOUT_INDEX_MAP,
                vis_vars=VIS_VARS,
                step=ctx.global_step,
            )
            self._figures_logged = True

        return metrics
