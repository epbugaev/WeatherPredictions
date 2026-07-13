"""``PredRNNStep`` — PredRNN-specific concat/permute around the forward pass.

Replaces ``LitModels.mutiout_predrnn``. PredRNN's API differs from the other
video predictors: input is ``(x, y)`` concatenated along the time axis and
permuted to channels-last; the second positional argument is the
mask-tensor used for scheduled sampling (we pass zeros, matching legacy).

The models return ``(next_frames, aux_losses)``, where ``aux_losses`` holds
already-weighted scalar penalties (PredRNN-V2's memory-decoupling term, and the
physics penalty of the PI variants). This strategy owns the task loss and adds
whatever auxiliary terms the model reports, logging each one separately so the
weight of a penalty against the task is visible.
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run a full ``(x, y)`` concat-permute-forward-permute pipeline.

    Args:
        model: A PredRNN-family model whose ``forward(frames, mask)`` returns
            ``(next_frames, aux_losses)``.
        x: Context clip, ``(B, T_ctx, C, H, W)``.
        y: Target clip, ``(B, T_pred, C, H, W)``.

    Returns:
        Tuple ``(inp, y_hat, aux_losses)``; ``inp``/``y_hat`` are
        ``(B, T, C, H, W)``, ``aux_losses`` maps a name to an already-weighted
        scalar penalty the caller must add to the task loss.
    """
    inp = torch.cat([x, y], dim=1)
    inp = inp.permute(0, 1, 3, 4, 2).contiguous()
    mask = torch.zeros((1, y.shape[1], 1, 1, 1), device=x.device, dtype=x.dtype)
    y_hat_perm, aux_losses = model(inp, mask)
    y_hat = y_hat_perm.permute(0, 1, 4, 2, 3)
    inp = inp.permute(0, 1, 4, 2, 3)
    return inp, y_hat, aux_losses


@register_strategy("mutiout_predrnn")
class PredRNNStep(StepStrategy):
    """PredRNN forward signature, model aux losses, per-variable RMSE, figures."""

    def __init__(self, log_figures_once: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_figures_once = log_figures_once
        self._figures_logged = False

    @staticmethod
    def _physics_residual_diagnostics(model: nn.Module) -> dict[str, torch.Tensor]:
        """Read the physics-branch diagnostics off a PI-model, if it exposes any.

        Duck-typed exactly like :class:`~training_strategies.iterative_manual.IterativeManualStep`
        so PI-PredRNNv2 (exp20) surfaces the same Comet panels as PI-IAM4VP —
        notably ``physics_residual_nonfinite_ratio``, without which a physics
        branch quietly degrading to its sanitize fallback is invisible. Plain
        PredRNN/PredRNNv2 have no such method and yield no keys.

        Args:
            model: the training model, possibly wrapped in ``DistributedDataParallel``.

        Returns:
            Mapping of diagnostic name to scalar ``torch.Tensor``; empty for
            models without a physics branch.
        """
        inner = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        diagnostics_fn = getattr(inner, "physics_residual_diagnostics", None)
        if diagnostics_fn is None:
            return {}
        return diagnostics_fn()

    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        inp, y_hat, aux_losses = _predrnn_forward(model, x, y)
        task_loss = self.loss(inp[:, 1:, ...], y_hat)
        loss = task_loss
        for aux_loss in aux_losses.values():
            loss = loss + aux_loss
        lr = torch.tensor(ctx.optimizer.param_groups[0]["lr"], device=loss.device)
        metrics = {"loss": loss, "task_loss": task_loss.detach(), "lr": lr}
        for name, aux_loss in aux_losses.items():
            metrics[f"aux_{name}"] = aux_loss.detach()
        metrics.update(self._physics_residual_diagnostics(model))
        return metrics

    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        _, y_hat_full, _ = _predrnn_forward(model, x, y)
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
