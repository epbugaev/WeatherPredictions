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
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

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

    def __init__(self, loss_type: str = "MAE", **_: Any) -> None:
        if loss_type == "MAE":
            self.loss = self._mae_loss
        elif loss_type == "MSE":
            self.loss = self._mse_loss
        else:
            raise ValueError(f"Unknown loss_type {loss_type!r}; expected MAE or MSE")
        self.loss_type = loss_type

    @staticmethod
    def _mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

    @staticmethod
    def _mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)

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
