"""``IterativeManualStep`` — IAM4VP-style per-timestep manual backward.

Replaces ``LitModels.mutiout_imvp`` and its byte-identical twin
``LitModels.mutiout_imvp_small_world``. The model is called once per
timestep with the running list of previous predictions; each per-timestep
loss is backwarded individually before a single ``optimizer.step`` at the
end of the batch.

``manual_optimization = True`` tells the trainer to skip its own
backward/step/scaler-update after ``train_step`` returns.
"""

from __future__ import annotations

import torch
from torch import nn

from training_strategies._index_maps import MULTIOUT_INDEX_MAP, VIS_VARS
from training_strategies._log_figures import log_prediction_maps
from training_strategies.base import StepContext, StepStrategy
from utils.registry import register_strategy


@register_strategy("mutiout_imvp")
@register_strategy("mutiout_imvp_small_world")
class IterativeManualStep(StepStrategy):
    """Per-timestep manual backward over ``time_prediction`` timesteps."""

    manual_optimization: bool = True

    def __init__(self, time_prediction: int = 6, log_figures_once: bool = True, **kwargs) -> None:
        """Configure the prediction horizon.

        Args:
            time_prediction: Number of timesteps to predict (and backward over).
            log_figures_once: Log prediction maps only on the first validation
                epoch (matches the legacy ``self.trained`` flag).
        """
        super().__init__(**kwargs)
        self.time_prediction = time_prediction
        self.log_figures_once = log_figures_once
        self._figures_logged = False

    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        """Iterate timesteps; manual backward per step; single step at the end.

        Mirrors the LitModels behaviour exactly: ``optimizer.zero_grad`` at the
        start, ``manual_backward`` after each per-timestep loss, single
        ``optimizer.step`` at the end. AMP is intentionally not wired in here
        because the legacy code did not use it; if AMP gets turned on later
        we'd wrap ``model(...)`` in ``autocast`` and use ``ctx.scaler``.
        """
        x, y = batch
        ctx.optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=ctx.device)
        pred_list: list[torch.Tensor] = []

        for idx_time in range(self.time_prediction):
            t = torch.tensor(
                (idx_time + 1) * 100, device=ctx.device
            ).repeat(x.shape[0])
            prediction = model(x, pred_list, t)
            pred_list.append(prediction.detach())

            step_loss = self.loss(prediction, y[:, idx_time])
            total_loss = total_loss + step_loss

            step_loss.backward()

        ctx.optimizer.step()

        avg_loss = total_loss / self.time_prediction
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=avg_loss.device)
        return {"loss": avg_loss, "lr": lr}

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch

        total_loss = torch.zeros((), device=ctx.device)
        pred_list: list[torch.Tensor] = []

        for idx_time in range(self.time_prediction):
            t = torch.tensor(
                (idx_time + 1) * 100, device=ctx.device
            ).repeat(x.shape[0])
            prediction = model(x, pred_list, t)
            pred_list.append(prediction.detach())
            total_loss = total_loss + self.loss(prediction, y[:, idx_time])

        val_loss = total_loss / self.time_prediction
        rmse_first = ctx.metrics.WRMSE(pred_list[0], y[:, 0])
        rmse_last = ctx.metrics.WRMSE(pred_list[-1], y[:, -1])

        metrics: dict[str, torch.Tensor] = {"val_loss": val_loss}
        for var_name, idx in MULTIOUT_INDEX_MAP.items():
            metrics[f"f RMSE_{var_name}_first"] = torch.as_tensor(
                rmse_first[idx], device=val_loss.device
            )
            metrics[f"f RMSE_{var_name}_last"] = torch.as_tensor(
                rmse_last[idx], device=val_loss.device
            )

        if ctx.is_main_process and (not self.log_figures_once or not self._figures_logged):
            log_prediction_maps(
                experiment=ctx.experiment,
                pred=pred_list[-1],
                truth=y[:, -1],
                index_map=MULTIOUT_INDEX_MAP,
                vis_vars=VIS_VARS,
                step=ctx.global_step,
            )
            self._figures_logged = True

        return metrics
