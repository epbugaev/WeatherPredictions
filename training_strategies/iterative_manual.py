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

    def _iterate_timesteps(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        ctx: StepContext,
        *,
        backward_each_step: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the per-timestep prediction loop, optionally backwarding each step.

        Args:
            model: predictor; called as ``model(x, pred_list, t)`` per step.
            x: input tensor.
            y: ground truth, shape ``(B, T, ...)``; the t-th step is supervised by ``y[:, t]``.
            ctx: trainer context (only ``device`` is read here).
            backward_each_step: when ``True``, calls ``step_loss.backward()`` after
                each per-timestep loss (manual_optimization train_step contract).

        Returns:
            ``(pred_list, total_loss)`` — predictions appended detached, summed loss.
        """
        total_loss = torch.zeros((), device=ctx.device)
        pred_list: list[torch.Tensor] = []
        for idx_time in range(self.time_prediction):
            t = torch.tensor((idx_time + 1) * 100, device=ctx.device).repeat(x.shape[0])
            prediction = model(x, pred_list, t)
            pred_list.append(prediction.detach())
            step_loss = self.loss(prediction, y[:, idx_time])
            total_loss = total_loss + step_loss
            if backward_each_step:
                step_loss.backward()
        return pred_list, total_loss

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

        _, total_loss = self._iterate_timesteps(model, x, y, ctx, backward_each_step=True)

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

        pred_list, total_loss = self._iterate_timesteps(model, x, y, ctx, backward_each_step=False)

        y_hat = torch.stack(pred_list, dim=1)
        val_loss = total_loss / self.time_prediction
        metrics = self._build_val_metrics(
            ctx,
            val_loss,
            pred_first=pred_list[0],
            target_first=y[:, 0],
            pred_last=pred_list[-1],
            target_last=y[:, -1],
            index_map=MULTIOUT_INDEX_MAP,
            pred_full=y_hat,
            target_full=y[:, : self.time_prediction],
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
