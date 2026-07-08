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

    @staticmethod
    def _inner_model(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    def _physics_residual_aux_loss(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        inner = self._inner_model(model)
        aux_fn = getattr(inner, "physics_residual_aux_loss", None)
        if aux_fn is None:
            return torch.zeros((), device=device)
        aux_loss = aux_fn()
        if aux_loss is None:
            return torch.zeros((), device=device)
        return aux_loss

    def _physics_residual_diagnostics(
        self,
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        inner = self._inner_model(model)
        diagnostics_fn = getattr(inner, "physics_residual_diagnostics", None)
        if diagnostics_fn is None:
            return {}
        return diagnostics_fn()

    def _set_residual_warmup(self, model: nn.Module, epoch: int) -> None:
        """Toggle IAM4VP-backbone freezing for the residual-corrector warmup.

        Args:
            model: predictor, possibly wrapped in ``DistributedDataParallel``.
            epoch: current epoch; the backbone is frozen while
                ``epoch < warmup_epochs`` and freezing is requested.

        Raises:
            ValueError: when freezing is requested with a positive warmup under a
                ``DistributedDataParallel`` wrapper. Toggling ``requires_grad``
                after DDP wrapping breaks gradient bucketing (parameters are
                bucketed once at construction time), so the warmup must be applied
                before the DDP wrap or the run must be single-GPU.
        """
        inner = self._inner_model(model)
        warmup_epochs = int(getattr(inner, "residual_warmup_epochs", 0))
        freeze = bool(getattr(inner, "freeze_iam4vp_for_residual_warmup", False))
        is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
        if freeze and warmup_epochs > 0 and is_ddp:
            raise ValueError(
                "freeze_iam4vp_for_residual_warmup=True with "
                f"residual_warmup_epochs={warmup_epochs} is incompatible with "
                "DistributedDataParallel: toggling requires_grad after DDP wrapping "
                "breaks gradient bucketing (parameters are bucketed at construction "
                "time). Apply the warmup before the DDP wrap or run single-GPU."
            )
        setter = getattr(inner, "set_residual_warmup", None)
        if setter is not None:
            setter(freeze and epoch < warmup_epochs)

    def _iterate_timesteps(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        ctx: StepContext,
        *,
        backward_each_step: bool,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Run the per-timestep prediction loop, optionally backwarding each step.

        Args:
            model: predictor; called as ``model(x, pred_list, t)`` per step.
            x: input tensor.
            y: ground truth, shape ``(B, T, ...)``; the t-th step is supervised by ``y[:, t]``.
            ctx: trainer context (only ``device`` is read here).
            backward_each_step: when ``True`` (train path) the physics aux loss is
                added to ``step_loss`` and ``step_loss.backward()`` is called after
                each per-timestep loss (manual_optimization train_step contract).
                When ``False`` (val path) the aux loss is still fetched and
                accumulated for logging but excluded from ``step_loss``, so
                ``val_loss`` / early-stopping semantics stay unchanged.

        Returns:
            Predictions appended detached, per-step loss (train: forecast + aux;
            val: forecast only), forecast-only loss, auxiliary residual loss
            (fetched in both train and val), and averaged residual diagnostics.
        """
        total_loss = torch.zeros((), device=ctx.device)
        forecast_total_loss = torch.zeros((), device=ctx.device)
        aux_total_loss = torch.zeros((), device=ctx.device)
        diagnostics: dict[str, torch.Tensor] = {}
        pred_list: list[torch.Tensor] = []
        for idx_time in range(self.time_prediction):
            t = torch.tensor((idx_time + 1) * 100, device=ctx.device).repeat(x.shape[0])
            prediction = model(x, pred_list, t)
            pred_list.append(prediction.detach())
            forecast_loss = self.loss(prediction, y[:, idx_time])
            aux_loss = self._physics_residual_aux_loss(model, ctx.device)
            step_loss = forecast_loss + aux_loss if backward_each_step else forecast_loss
            forecast_total_loss = forecast_total_loss + forecast_loss.detach()
            aux_total_loss = aux_total_loss + aux_loss.detach()
            total_loss = total_loss + step_loss
            for key, value in self._physics_residual_diagnostics(model).items():
                diagnostics[key] = diagnostics.get(key, torch.zeros((), device=ctx.device)) + value
            if backward_each_step:
                step_loss.backward()
        if diagnostics:
            diagnostics = {key: value / self.time_prediction for key, value in diagnostics.items()}
        return pred_list, total_loss, forecast_total_loss, aux_total_loss, diagnostics

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
        self._set_residual_warmup(model, ctx.epoch)
        ctx.optimizer.zero_grad(set_to_none=True)

        _, total_loss, forecast_total_loss, aux_total_loss, diagnostics = self._iterate_timesteps(
            model, x, y, ctx, backward_each_step=True
        )

        ctx.optimizer.step()

        avg_loss = total_loss / self.time_prediction
        avg_forecast_loss = forecast_total_loss / self.time_prediction
        avg_aux_loss = aux_total_loss / self.time_prediction
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=avg_loss.device)
        metrics = {
            "loss": avg_loss.detach(),
            "forecast_loss": avg_forecast_loss,
            "physics_residual_aux_loss": avg_aux_loss,
            "lr": lr,
        }
        metrics.update(diagnostics)
        return metrics

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch

        pred_list, total_loss, _, aux_total_loss, diagnostics = self._iterate_timesteps(
            model, x, y, ctx, backward_each_step=False
        )

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
        metrics.update({f"val_{key}": value for key, value in diagnostics.items()})
        metrics["val_physics_residual_aux_loss"] = aux_total_loss / self.time_prediction

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
