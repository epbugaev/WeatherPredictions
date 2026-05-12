"""Pure-PyTorch training loop.

The trainer owns the per-epoch loop, DDP wrapping, AMP autocast/grad-scaler,
scheduler stepping, periodic logging, checkpoint writes and early stopping.
Per-batch logic (forward, loss, manual backward when needed) is delegated to
a ``StepStrategy``; metrics aggregation is done in-place at the trainer
level.

The scheduler is stepped once per optimiser step (per-batch ``CosineAnnealingLR``
schedule, matching the previous training configuration).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training_strategies.base import StepContext, StepStrategy
from utils.checkpointing import save_checkpoint
from utils.distributed import DistInfo, barrier, reduce_dict
from utils.early_stopping import EarlyStoppingTracker
from utils.experiment import Experiment
from utils.metrics import Metrics

# Sentinel returned by ``next(iter, default)`` when the loader is exhausted.
# Used in the training loop to mark end-of-iteration without try/except (LBYL).
_END_OF_LOADER = object()


def _precision_to_amp(precision: int | str) -> tuple[bool, torch.dtype]:
    """Translate the YAML ``trainer.precision`` value into AMP settings.

    Args:
        precision: ``32`` (no AMP), ``16``/``"16-mixed"`` (fp16 AMP), or
            ``"bf16"``/``"bf16-mixed"`` (bf16 AMP).

    Returns:
        ``(amp_enabled, amp_dtype)``.
    """
    if isinstance(precision, str):
        precision = precision.lower()
    if precision in (32, "32", "32-true"):
        return False, torch.float32
    if precision in (16, "16", "16-mixed"):
        return True, torch.float16
    if precision in ("bf16", "bf16-mixed"):
        return True, torch.bfloat16
    raise ValueError(f"Unsupported precision setting: {precision!r}")


@dataclass
class TrainerConfig:
    """Static configuration consumed by ``Trainer``.

    Attributes:
        max_epochs: Maximum number of training epochs.
        precision: ``32`` / ``16`` / ``"bf16"`` — controls AMP autocast.
        log_every_n_steps: Period (in batches) for logging training metrics.
        static_graph: Passed to ``DistributedDataParallel`` for static graphs.
        find_unused_parameters: DDP setting; defaults to False, which is
            the right choice with ``static_graph=True``.
        early_stopping_patience: Patience for the val-loss tracker.
        checkpoint_dir: Directory where ``best.pt`` and ``last.pt`` are written.
        float32_matmul_precision: Optional ``torch.set_float32_matmul_precision``
            value (e.g. ``"high"``). ``None`` leaves the default.
    """

    max_epochs: int
    precision: int | str = 32
    log_every_n_steps: int = 5
    static_graph: bool = True
    find_unused_parameters: bool = False
    early_stopping_patience: int = 5
    checkpoint_dir: str = "./checkpoints"
    float32_matmul_precision: str | None = None


class Trainer:
    """Drives the train/val loop using a ``StepStrategy``."""

    def __init__(self, cfg: TrainerConfig, dist_info: DistInfo) -> None:
        """Build the trainer.

        Args:
            cfg: Static trainer settings derived from YAML.
            dist_info: Distributed runtime info from ``setup_distributed``.
        """
        self.cfg = cfg
        self.dist = dist_info
        self.amp_enabled, self.amp_dtype = _precision_to_amp(cfg.precision)

        if cfg.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # -------- public API --------

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        strategy: StepStrategy,
        experiment: Experiment,
        metrics: Metrics,
        full_config: dict[str, Any],
    ) -> None:
        """Train the model.

        Args:
            model: ``nn.Module`` (raw, not DDP-wrapped).
            train_loader: Training DataLoader; if its sampler is a
                ``DistributedSampler``, the trainer calls ``set_epoch`` each
                epoch.
            val_loader: Validation DataLoader.
            optimizer: Optimiser to update model parameters.
            scheduler: LR scheduler; stepped once per optimiser step.
            strategy: Per-batch strategy (forward, loss, optional backward).
            experiment: Experiment facade for metrics/figure logging.
            metrics: Domain metrics helper (used by strategies through the
                ``StepContext``).
            full_config: YAML config dict, persisted with each checkpoint.
        """
        model = model.to(self.dist.device)
        if self.dist.is_distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist.local_rank] if torch.cuda.is_available() else None,
                static_graph=self.cfg.static_graph,
                find_unused_parameters=self.cfg.find_unused_parameters,
            )

        scaler = GradScaler(enabled=self.amp_enabled)
        early_stopping = EarlyStoppingTracker(
            patience=self.cfg.early_stopping_patience,
            mode="min",
            check_finite=True,
        )

        global_step = 0
        for epoch in range(self.cfg.max_epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            global_step = self._train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                strategy=strategy,
                experiment=experiment,
                metrics=metrics,
                epoch=epoch,
                global_step=global_step,
            )

            val_metrics = self._validate(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                strategy=strategy,
                experiment=experiment,
                metrics=metrics,
                epoch=epoch,
                global_step=global_step,
            )

            val_loss = float(val_metrics["val_loss"].item())
            improved = early_stopping.is_improvement(val_loss)

            if self.dist.rank == 0:
                self._write_checkpoints(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if self.amp_enabled else None,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                    config=full_config,
                    improved=improved,
                )

            if early_stopping.update(val_loss):
                if self.dist.rank == 0:
                    print(f"[trainer] early stop at epoch {epoch} (val_loss={val_loss:.6f})")
                break

            barrier(self.dist)

        if self.dist.rank == 0:
            print("[trainer] training finished")

    # -------- private helpers --------

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        strategy: StepStrategy,
        experiment: Experiment,
        metrics: Metrics,
        epoch: int,
        global_step: int,
    ) -> int:
        """Run one training epoch; returns the new ``global_step``.

        The loop is instrumented with ``torch.profiler.record_function`` tags
        (``data_wait``, ``to_device``, ``forward``, ``backward``,
        ``optimizer_step``, ``scheduler_step``, ``log_metrics``) to make
        per-phase timing visible in profiler traces. Tags are no-ops when
        the profiler is disabled.
        """
        model.train()
        loader_iter = iter(loader)
        while True:
            with record_function("data_wait"):
                batch = next(loader_iter, _END_OF_LOADER)
            if batch is _END_OF_LOADER:
                break

            with record_function("to_device"):
                batch = _to_device(batch, self.dist.device)
            ctx = StepContext(
                device=self.dist.device,
                optimizer=optimizer,
                scaler=scaler,
                experiment=experiment,
                metrics=metrics,
                global_step=global_step,
                epoch=epoch,
                is_main_process=self.dist.rank == 0,
            )

            if strategy.manual_optimization:
                with record_function("train_step_manual"):
                    step_metrics = strategy.train_step(model, batch, ctx)
            else:
                with record_function("zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with record_function("forward"):
                    with autocast(
                        device_type=self.dist.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_enabled,
                    ):
                        step_metrics = strategy.train_step(model, batch, ctx)
                with record_function("backward"):
                    loss = step_metrics["loss"]
                    scaler.scale(loss).backward()
                with record_function("optimizer_step"):
                    scaler.step(optimizer)
                    scaler.update()

            with record_function("scheduler_step"):
                scheduler.step()
            global_step += 1

            if global_step % self.cfg.log_every_n_steps == 0:
                with record_function("log_metrics"):
                    self._log_train_metrics(step_metrics, experiment, global_step)

        return global_step

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        strategy: StepStrategy,
        experiment: Experiment,
        metrics: Metrics,
        epoch: int,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        """Run validation over ``loader`` and return reduced epoch metrics."""
        model.eval()
        accum: dict[str, torch.Tensor] = {}
        count = 0
        for batch in loader:
            batch = _to_device(batch, self.dist.device)
            ctx = StepContext(
                device=self.dist.device,
                optimizer=optimizer,
                scaler=scaler,
                experiment=experiment,
                metrics=metrics,
                global_step=global_step,
                epoch=epoch,
                is_main_process=self.dist.rank == 0,
            )
            with autocast(
                device_type=self.dist.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                step_metrics = strategy.val_step(model, batch, ctx)
            for key, value in step_metrics.items():
                detached = value.detach().to(self.dist.device).float()
                accum[key] = accum.get(key, torch.zeros((), device=self.dist.device)) + detached
            count += 1

        if count == 0:
            return {"val_loss": torch.tensor(float("inf"), device=self.dist.device)}

        averaged = {k: v / count for k, v in accum.items()}
        reduced = reduce_dict(averaged, self.dist, average=True)

        if self.dist.rank == 0:
            experiment.log_metrics(
                {k: float(v.item()) for k, v in reduced.items()}, step=global_step
            )

        return reduced

    def _log_train_metrics(
        self,
        step_metrics: dict[str, torch.Tensor],
        experiment: Experiment,
        global_step: int,
    ) -> None:
        """Reduce per-batch metrics and log on rank 0."""
        detached = {
            k: v.detach().to(self.dist.device).float() for k, v in step_metrics.items()
        }
        reduced = reduce_dict(detached, self.dist, average=True)
        if self.dist.rank == 0:
            payload = {
                ("train_loss" if k == "loss" else k): float(v.item()) for k, v in reduced.items()
            }
            experiment.log_metrics(payload, step=global_step)

    def _write_checkpoints(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler | None,
        epoch: int,
        global_step: int,
        val_loss: float,
        config: dict[str, Any],
        improved: bool,
    ) -> None:
        """Save ``last.pt`` and, when ``improved``, ``best.pt`` and a tagged copy."""
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        last_path = os.path.join(self.cfg.checkpoint_dir, "last.pt")
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            metric=val_loss,
            config=config,
        )
        if improved:
            best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metric=val_loss,
                config=config,
            )
            tagged_path = os.path.join(
                self.cfg.checkpoint_dir, f"epoch={epoch:02d}-val_loss={val_loss:.4f}.pt"
            )
            save_checkpoint(
                path=tagged_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metric=val_loss,
                config=config,
            )


def _to_device(batch: Any, device: torch.device) -> Any:
    """Move tensors in a batch to ``device``; tuples/lists are recursed into."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (tuple, list)):
        moved = [_to_device(item, device) for item in batch]
        return type(batch)(moved)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch
