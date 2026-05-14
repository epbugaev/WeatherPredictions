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

import atexit
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training_strategies.base import StepContext, StepStrategy
from utils.distributed import DistInfo, barrier, reduce_dict
from utils.early_stopping import EarlyStoppingTracker
from utils.experiment import Experiment
from utils.metrics import Metrics

# Sentinel returned by ``next(iter, default)`` when the loader is exhausted.
# Used in the training loop to mark end-of-iteration without try/except (LBYL).
_END_OF_LOADER = object()

# Bench-mode env vars, exported by ``sh_files/bench_dataloader.sh``:
#   BENCH_MAX_STEPS:  if > 0, the training loop breaks after this many
#                     optimisation steps and ``fit()`` skips validation and
#                     checkpoint writes.
#   PROFILE_TRACE_DIR: if set, the training loop is wrapped in
#                     ``torch.profiler.profile``; the chrome-trace JSON is
#                     written to this directory after warmup+active steps.
# Both default unset, leaving production behaviour unchanged.
_BENCH_MAX_STEPS = int(os.environ.get("BENCH_MAX_STEPS", "0"))
_PROFILE_TRACE_DIR = os.environ.get("PROFILE_TRACE_DIR") or None
_BENCH_MODE = bool(_BENCH_MAX_STEPS) or bool(_PROFILE_TRACE_DIR)


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
        grad_clip_norm: If set, clip the global gradient norm to this value
            before ``optimizer.step()``. Defends against gradient explosion
            under bf16/fp16 AMP. ``None`` disables clipping.
        skip_non_finite_loss: When True, batches whose forward produces a
            ``NaN``/``Inf`` loss are skipped (no backward, no optimizer
            step). The scheduler still advances so the cosine schedule
            stays aligned across ranks. A warning is printed from rank 0.
    """

    max_epochs: int
    precision: int | str = 32
    log_every_n_steps: int = 5
    static_graph: bool = True
    find_unused_parameters: bool = False
    early_stopping_patience: int = 5
    checkpoint_dir: str = "./checkpoints"
    float32_matmul_precision: str | None = None
    grad_clip_norm: float | None = None
    skip_non_finite_loss: bool = True
    # Run validation (and the rank-0 checkpoint write that depends on it)
    # only every Nth epoch. ``1`` = legacy behaviour. At larger values the
    # ``early_stopping_patience`` should be scaled down by the same factor
    # so the "epochs of plateau" semantics stay equivalent.
    val_every_n_epochs: int = 1


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

        # Background checkpoint writer (rank 0 only). The training loop
        # snapshots state_dicts synchronously (~1-3 s on CPU) and submits
        # the actual ``torch.save`` + ``os.replace`` to this pool, so the
        # DDP barrier after _write_checkpoints isn't blocked on disk I/O.
        # ``atexit`` waits for the last in-flight write before the process
        # exits so we never leave a half-written ``.tmp`` file behind.
        self._ckpt_pool: ThreadPoolExecutor | None = None
        self._pending_ckpt: Future | None = None
        if self.dist.rank == 0:
            self._ckpt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt_writer")
            atexit.register(self._drain_pending_ckpt)

    def _drain_pending_ckpt(self) -> None:
        """Wait for the in-flight checkpoint write (if any) and shut the pool."""
        if self._pending_ckpt is not None:
            self._pending_ckpt.result()
            self._pending_ckpt = None
        if self._ckpt_pool is not None:
            self._ckpt_pool.shutdown(wait=True)
            self._ckpt_pool = None

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
        normalize: nn.Module | None = None,
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
            normalize: Optional pre-model module (e.g.
                ``utils.normalize.WeatherNormalize``). When provided, it is
                moved to the trainer's device once and applied to every batch
                right after ``_to_device`` so strategies/models always see
                normalized tensors. Required for v4 datasets that return raw
                memmap values.
        """
        model = model.to(self.dist.device)
        if self.dist.is_distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist.local_rank] if torch.cuda.is_available() else None,
                static_graph=self.cfg.static_graph,
                find_unused_parameters=self.cfg.find_unused_parameters,
            )

        self._normalize = normalize.to(self.dist.device) if normalize is not None else None

        # GradScaler is only meaningful for fp16 AMP — bf16 has fp32's dynamic
        # range and never overflows, so no loss scaling is needed. Keeping it
        # enabled for bf16 adds per-step overhead from no-op scale/unscale calls.
        scaler = GradScaler(enabled=(self.amp_enabled and self.amp_dtype == torch.float16))
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

            if _BENCH_MODE:
                if self.dist.rank == 0:
                    print(
                        f"[bench] mode active "
                        f"(BENCH_MAX_STEPS={_BENCH_MAX_STEPS}, "
                        f"PROFILE_TRACE_DIR={_PROFILE_TRACE_DIR}); "
                        f"skipping validation and checkpointing"
                    )
                break

            # Validate every ``val_every_n_epochs`` (and on the last epoch).
            # On skipped epochs we save no checkpoint and don't update
            # early-stopping — the gap is "free" GPU time vs the ~30-60 sec
            # validation + checkpoint sync per epoch.
            should_validate = (epoch + 1) % self.cfg.val_every_n_epochs == 0 or (
                epoch + 1
            ) == self.cfg.max_epochs
            if should_validate:
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

        Bench-mode shortcuts (driven by ``BENCH_MAX_STEPS`` /
        ``PROFILE_TRACE_DIR`` env vars; both unset in production):

        * if ``PROFILE_TRACE_DIR`` is set, the whole loop runs inside a
          ``torch.profiler.profile`` context that dumps a Chrome-tracing
          JSON to that directory after warmup+active iterations;
        * if ``BENCH_MAX_STEPS`` is set, the loop breaks after that many
          optimisation steps (no extra epochs, no validation — see
          ``fit()`` for the rank-0 short-circuit).
        """
        model.train()

        prof_ctx = nullcontext()
        if _PROFILE_TRACE_DIR:
            prof_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=10, active=40, repeat=1),
                on_trace_ready=tensorboard_trace_handler(_PROFILE_TRACE_DIR),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )

        with prof_ctx as prof:
            loader_iter = iter(loader)
            while True:
                with record_function("data_wait"):
                    batch = next(loader_iter, _END_OF_LOADER)
                if batch is _END_OF_LOADER:
                    break

                with record_function("to_device"):
                    batch = _to_device(batch, self.dist.device)
                if self._normalize is not None:
                    with record_function("normalize"):
                        batch = _normalize_batch(batch, self._normalize)
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
                    loss = step_metrics["loss"]
                    # NaN/Inf guard: under bf16/fp16 a single bad sample or
                    # exploding gradient can poison the optimiser state for
                    # the rest of training. Detect, skip backward+step, but
                    # still advance the scheduler so cosine stays aligned
                    # across ranks. All ranks see the same `loss` (same data
                    # rank-locally), so no cross-rank all_reduce needed here.
                    if self.cfg.skip_non_finite_loss and not torch.isfinite(loss):
                        if self.dist.rank == 0:
                            print(
                                f"[trainer] step {global_step}: non-finite loss "
                                f"{loss.item()!r}, skipping backward/optimizer.step"
                            )
                        with record_function("scheduler_step"):
                            scheduler.step()
                        global_step += 1
                        if prof is not None:
                            prof.step()
                        if _BENCH_MAX_STEPS and global_step >= _BENCH_MAX_STEPS:
                            break
                        continue
                    with record_function("backward"):
                        scaler.scale(loss).backward()
                    if self.cfg.grad_clip_norm is not None:
                        with record_function("clip_grad"):
                            # ``unscale_`` is a no-op when the scaler is
                            # disabled (bf16), and unscales gradients to the
                            # real scale when it's enabled (fp16) so clipping
                            # sees pre-scaled values.
                            scaler.unscale_(optimizer)
                            inner_model = (
                                model.module
                                if isinstance(model, nn.parallel.DistributedDataParallel)
                                else model
                            )
                            torch.nn.utils.clip_grad_norm_(
                                inner_model.parameters(),
                                max_norm=self.cfg.grad_clip_norm,
                            )
                    with record_function("optimizer_step"):
                        scaler.step(optimizer)
                        scaler.update()

                with record_function("scheduler_step"):
                    scheduler.step()
                global_step += 1

                if prof is not None:
                    prof.step()

                if _BENCH_MAX_STEPS and global_step >= _BENCH_MAX_STEPS:
                    break

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
            if self._normalize is not None:
                batch = _normalize_batch(batch, self._normalize)
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
        """Log rank-0 training metrics without cross-rank ``all_reduce``.

        Train-loss is logged from rank 0's local values rather than the
        DDP-averaged value. DistributedSampler shards data across ranks,
        so rank-0 loss is an unbiased estimator on its own shard — the
        per-step noise is marginally higher, but eliminating the
        all_reduce + Comet I/O round-trip every ``log_every_n_steps``
        removes a per-50-step idle hole on rank 1. Validation still
        reduces (val_loss feeds early-stopping and must be exact).
        """
        if self.dist.rank != 0:
            return
        payload = {
            ("train_loss" if k == "loss" else k): float(v.detach().item())
            for k, v in step_metrics.items()
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
        """Save ``last.pt`` and, when ``improved``, ``best.pt`` and a tagged copy.

        Strategy: snapshot the state-dicts on CPU synchronously (~1-3 s
        — this is what blocks rank 0), then offload the actual
        ``torch.save`` + ``os.replace`` to a background thread so the
        DDP barrier following this method doesn't wait for disk I/O.
        """
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # Wait on the previous epoch's write before we overwrite its
        # paths. In steady state this is instant (the write finished
        # while the next epoch was training).
        if self._pending_ckpt is not None:
            self._pending_ckpt.result()
            self._pending_ckpt = None

        payload = _take_checkpoint_snapshot(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            metric=val_loss,
            config=config,
        )

        last_path = os.path.join(self.cfg.checkpoint_dir, "last.pt")
        paths: list[str] = [last_path]
        if improved:
            paths.append(os.path.join(self.cfg.checkpoint_dir, "best.pt"))
            paths.append(
                os.path.join(
                    self.cfg.checkpoint_dir,
                    f"epoch={epoch:02d}-val_loss={val_loss:.4f}.pt",
                )
            )

        # All three paths share the same payload (single CPU snapshot),
        # so the background thread does N torch.save calls in sequence.
        assert self._ckpt_pool is not None
        self._pending_ckpt = self._ckpt_pool.submit(_write_checkpoint_payload, payload, paths)


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


def _normalize_batch(batch: Any, normalize: nn.Module) -> Any:
    """Apply ``normalize`` to all tensors in ``batch``, preserving structure.

    Used by v4 datasets that return raw memmap rows: the trainer moves the
    batch to device then z-scores it once before strategies/models see it.
    """
    if isinstance(batch, torch.Tensor):
        return normalize(batch)
    if isinstance(batch, (tuple, list)):
        normed = [_normalize_batch(item, normalize) for item in batch]
        return type(batch)(normed)
    if isinstance(batch, dict):
        return {k: _normalize_batch(v, normalize) for k, v in batch.items()}
    return batch


def _take_checkpoint_snapshot(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler | None,
    epoch: int,
    global_step: int,
    metric: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a CPU-resident checkpoint payload that's safe to hand to a thread.

    Runs in the training thread. The optimiser / scheduler / scaler
    ``state_dict()`` methods already return CPU-resident, deep-enough
    copies; only the model weights live on the GPU, so those we clone
    explicitly to CPU. After this returns, the training loop can mutate
    the live state freely without affecting the background writer.
    """
    raw = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    payload: dict[str, Any] = {
        "model": {k: v.detach().to("cpu", copy=True) for k, v in raw.state_dict().items()},
        "epoch": epoch,
        "global_step": global_step,
        "metric": metric,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def _write_checkpoint_payload(payload: dict[str, Any], paths: list[str]) -> None:
    """Write ``payload`` to every path in ``paths`` via tmp-file + atomic rename.

    Runs in a background thread. Each path gets the same on-disk bytes,
    so we only serialise once and ``shutil.copy`` the file for additional
    destinations — but ``torch.save`` is fast enough on CPU snapshots
    (~5-15 s for 100-500 MB) that the simpler ``save-then-copy`` pattern
    isn't worth the complexity. Sequential ``torch.save`` it is.
    """
    for path in paths:
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
