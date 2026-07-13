"""``StepStrategy`` interface for per-batch train/val logic.

A strategy encapsulates only the per-batch behaviour (forward, loss, metrics,
optional manual backward); the training loop, optimiser, scheduler, AMP and
DDP all live in ``trainer.Trainer`` and are passed in through ``StepContext``.

A concrete strategy implements:
  * ``train_step(model, batch, ctx)`` — returns a metrics dict containing at
    least ``loss`` (when ``manual_optimization == False``).
  * ``val_step(model, batch, ctx)`` — returns a metrics dict containing at
    least ``val_loss``.

When ``manual_optimization`` is ``True`` the strategy owns the backward and
optimiser calls; the trainer will skip its own ``backward``/``step`` after
``train_step`` returns. This is the path used by IAM4VP-style per-timestep
manual backward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from training_strategies._index_maps import DEFAULT_VALIDATION_CHANNELS
from utils.experiment import Experiment
from utils.metrics import Metrics


@dataclass
class StepContext:
    """Per-batch context that the trainer hands to strategies.

    Attributes:
        device: Device on which the batch tensors and model live.
        optimizer: Optimiser, needed by strategies with ``manual_optimization``.
        scaler: AMP grad scaler (no-op when AMP is disabled).
        experiment: Logger facade for figures; metrics are returned via the
            strategy's return value, not logged from inside the strategy.
        metrics: Domain-specific metrics helper (weighted RMSE etc.).
        global_step: Current optimiser-step counter, useful for figure naming.
        epoch: Current epoch index (0-based).
        is_main_process: True only on global rank 0; strategies should gate
            figure logging on this flag.
    """

    device: torch.device
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    experiment: Experiment
    metrics: Metrics
    global_step: int
    epoch: int
    is_main_process: bool


class StepStrategy(ABC):
    """Abstract per-batch train/val behaviour.

    Subclasses set ``manual_optimization`` to ``True`` when they perform their
    own ``loss.backward()`` and ``optimizer.step()`` (e.g. IAM4VP-style
    per-timestep manual backward). The default ``False`` means automatic
    optimisation: the strategy returns the loss in the metrics dict and the
    trainer performs backward + step.
    """

    manual_optimization: bool = False

    def __init__(
        self,
        loss_type: str = "MAE",
        validation_channels: Sequence[str] | str | None = None,
        **_: Any,
    ) -> None:
        if loss_type == "MAE":
            self.loss = self._mae_loss
        elif loss_type == "MSE":
            self.loss = self._mse_loss
        else:
            raise ValueError(f"Unknown loss_type {loss_type!r}; expected MAE or MSE")
        self.loss_type = loss_type
        self.validation_channels = self._coerce_validation_channels(validation_channels)

    @staticmethod
    def _coerce_validation_channels(
        validation_channels: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        """Normalize optional config-driven validation channel names.

        ``None`` keeps the default exact 69-channel WeatherBench layout. A
        comma-separated string is accepted so users can override from YAML/env
        plumbing without changing code.
        """
        if validation_channels is None:
            return DEFAULT_VALIDATION_CHANNELS
        if isinstance(validation_channels, str):
            return tuple(part.strip() for part in validation_channels.split(",") if part.strip())
        return tuple(validation_channels)

    @staticmethod
    def _inner_model(model: nn.Module) -> nn.Module:
        """Unwrap ``DistributedDataParallel`` so duck-typed hooks stay visible.

        Args:
            model: the training model, possibly DDP-wrapped.

        Returns:
            The underlying module (the argument itself when not wrapped).
        """
        return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    def _physics_residual_aux_loss(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Read the physics-branch auxiliary penalty off a PI-model.

        Duck-typed on :class:`~utils.physics_residual.PhysicsResidualMixin`: any
        model exposing ``physics_residual_aux_loss`` contributes its L1 penalty
        on the residual correction, which the caller must add to the task loss.
        Models without a physics branch yield a zero scalar, so every strategy
        can call this unconditionally and stay bit-exact for them.

        Args:
            model: the training model, possibly DDP-wrapped.
            device: device for the zero fallback.

        Returns:
            Scalar ``torch.Tensor``; zero for models without a physics branch.
        """
        aux_fn = getattr(self._inner_model(model), "physics_residual_aux_loss", None)
        if aux_fn is None:
            return torch.zeros((), device=device)
        aux_loss = aux_fn()
        if aux_loss is None:
            return torch.zeros((), device=device)
        return aux_loss

    def _physics_residual_diagnostics(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Read the physics-branch diagnostics off a PI-model, if it exposes any.

        These carry ``physics_residual_nonfinite_ratio``, without which a physics
        branch quietly degrading to its sanitize fallback would be invisible in
        Comet.

        Args:
            model: the training model, possibly DDP-wrapped.

        Returns:
            Mapping of diagnostic name to scalar ``torch.Tensor``; empty for
            models without a physics branch.
        """
        diagnostics_fn = getattr(self._inner_model(model), "physics_residual_diagnostics", None)
        if diagnostics_fn is None:
            return {}
        return diagnostics_fn()

    @staticmethod
    def _mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

    @staticmethod
    def _mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)

    @staticmethod
    def _per_variable_rmse_metrics(
        rmse_first: list[float] | torch.Tensor,
        rmse_last: list[float] | torch.Tensor,
        index_map: dict[str, int],
        device: torch.device,
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Собрать словарь `{prefix}RMSE_<var>_first/last` из per-channel WRMSE.

        Args:
            rmse_first: per-channel WRMSE на первом таймстепе (1-D iterable).
            rmse_last: per-channel WRMSE на последнем таймстепе.
            index_map: имя переменной → индекс канала в WRMSE-векторе.
            device: устройство, на которое преобразовать каждую скалярную метрику.
            prefix: опциональный префикс ключей метрик; по умолчанию пустой,
                все текущие стратегии логируют под общим именем ``RMSE_<var>_*``.

        Returns:
            Словарь скалярных 0-d тензоров для агрегации в трейнере.
        """
        metrics: dict[str, torch.Tensor] = {}
        for var_name, idx in index_map.items():
            metrics[f"{prefix}RMSE_{var_name}_first"] = torch.as_tensor(
                rmse_first[idx], device=device
            )
            metrics[f"{prefix}RMSE_{var_name}_last"] = torch.as_tensor(
                rmse_last[idx], device=device
            )
        return metrics

    def _horizon_mean_rmse_metrics(
        self,
        ctx: StepContext,
        pred: torch.Tensor,
        target: torch.Tensor,
        index_map: dict[str, int],
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Build `{prefix}RMSE_<var>_mean` from all validation target timesteps.

        The channel labels must be exact entries in ``index_map``. For example,
        ``t500`` is valid but ``t450`` is not present in the 69-channel
        WeatherBench layout unless a separate interpolation metric is added.
        """
        if not self.validation_channels:
            return {}

        missing = [name for name in self.validation_channels if name not in index_map]
        if missing:
            available = ", ".join(index_map)
            raise ValueError(
                "Unknown validation channel(s): "
                f"{', '.join(missing)}. Available exact channels: {available}"
            )

        rmse = torch.as_tensor(ctx.metrics.WRMSE(pred, target), device=target.device)
        if rmse.dim() == 1:
            rmse_by_channel = rmse
        elif rmse.dim() == 2:
            rmse_by_channel = rmse.mean(dim=0)
        else:
            raise ValueError(
                "WRMSE must return shape (C,) for 4D inputs or (T, C) for 5D inputs, "
                f"got {tuple(rmse.shape)}"
            )

        return {
            f"{prefix}RMSE_{var_name}_mean": rmse_by_channel[index_map[var_name]]
            for var_name in self.validation_channels
        }

    def _build_val_metrics(
        self,
        ctx: StepContext,
        val_loss: torch.Tensor,
        pred_first: torch.Tensor,
        target_first: torch.Tensor,
        pred_last: torch.Tensor,
        target_last: torch.Tensor,
        index_map: dict[str, int],
        prefix: str = "",
        pred_full: torch.Tensor | None = None,
        target_full: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Собрать val-словарь: ``val_loss`` + per-variable RMSE.

        Args:
            ctx: StepContext (используется ``ctx.metrics.WRMSE``).
            val_loss: уже вычисленный скалярный loss валидации.
            pred_first: предсказание на первом таймстепе (срез вида ``y_hat[:, 0]``).
            target_first: эталон на первом таймстепе.
            pred_last: предсказание на последнем таймстепе.
            target_last: эталон на последнем таймстепе.
            index_map: имя переменной → индекс канала в WRMSE-векторе.
            prefix: опциональный префикс ключей; по умолчанию пустой — единый
                namespace метрик для всех моделей (нужно для сравнения в Comet).
            pred_full: весь прогноз ``(B, T, C, H, W)`` для horizon-mean RMSE.
            target_full: весь target той же формы.

        Returns:
            ``{"val_loss": ..., "{prefix}RMSE_<var>_first/last/mean": ...}``.
        """
        rmse_first = ctx.metrics.WRMSE(pred_first, target_first)
        rmse_last = ctx.metrics.WRMSE(pred_last, target_last)
        metrics: dict[str, torch.Tensor] = {"val_loss": val_loss}
        metrics.update(
            self._per_variable_rmse_metrics(
                rmse_first, rmse_last, index_map, val_loss.device, prefix=prefix
            )
        )
        if pred_full is not None and target_full is not None:
            metrics.update(
                self._horizon_mean_rmse_metrics(
                    ctx, pred_full, target_full, index_map, prefix=prefix
                )
            )
        return metrics

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        """Run one training step on ``batch`` and return scalar metrics.

        Args:
            model: Forward-able model (raw ``nn.Module`` or DDP-wrapped).
            batch: A single batch yielded by the train DataLoader.
            ctx: Per-step context with device, optimiser, scaler, etc.

        Returns:
            Dict of scalar 0-dim tensors. Must include ``loss`` when
            ``self.manual_optimization`` is ``False``.
        """

    @abstractmethod
    def val_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, ...],
        ctx: StepContext,
    ) -> dict[str, torch.Tensor]:
        """Run one validation step on ``batch`` and return scalar metrics.

        Args:
            model: Forward-able model in eval mode.
            batch: A single batch yielded by the validation DataLoader.
            ctx: Per-step context.

        Returns:
            Dict of scalar 0-dim tensors. Must include ``val_loss``.
        """
