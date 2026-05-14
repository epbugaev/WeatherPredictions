"""Shared boilerplate for every specialised ``train/train_<Model>.py``.

Each script in this package handles its own model / dataset construction
(historical experiments hardcode shapes, cuts and date ranges) and then
calls :func:`run_legacy_training` which performs the bits every script
shares: distributed setup, DataLoaders with optional ``DistributedSampler``,
optimiser+scheduler, ``Trainer`` configuration, Comet experiment, checkpoint
directory, the actual ``fit`` call, and tear-down.
"""

from __future__ import annotations

import datetime
import os
import random
import string
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import Data  # noqa: F401
import Models  # noqa: F401
import training_strategies  # noqa: F401
from trainer import Trainer, TrainerConfig
from utils.distributed import cleanup_distributed, setup_distributed
from utils.experiment import build_experiment
from utils.metrics import Metrics
from utils.registry import get_strategy


def _make_run_id() -> str:
    """Compose the run-id used as the checkpoint subdir name."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=5)
    )


def run_legacy_training(
    *,
    model: torch.nn.Module,
    train_data: Dataset,
    valid_data: Dataset,
    exp_name: str,
    strategy_name: str,
    strategy_kwargs: dict[str, Any] | None = None,
    train_loader_kwargs: dict[str, Any] | None = None,
    val_loader_kwargs: dict[str, Any] | None = None,
    lr: float = 5e-4,
    eta_min: float = 0.0,
    max_epoch: int = 20,
    loss_type: str = "MAE",
    early_stopping_patience: int = 5,
    precision: int | str = 32,
    log_every_n_steps: int = 5,
    static_graph: bool = True,
    checkpoint_base: str = "/home/ebugaev/checkpoints/",
    comet_api_key: str = "",
    comet_project: str = "WeatherPredictions",
    log_code_file: str | None = None,
    metrics_source: Dataset | None = None,
) -> None:
    """Wire the supplied components into a ``Trainer`` and run ``fit``.

    Args:
        model: ``nn.Module`` to train.
        train_data: Training ``Dataset``.
        valid_data: Validation ``Dataset``.
        exp_name: Comet experiment name and checkpoint subdir prefix.
        strategy_name: Registry key for the ``StepStrategy``.
        strategy_kwargs: Extra kwargs forwarded to the strategy constructor
            (e.g. ``time_prediction``).
        train_loader_kwargs: ``DataLoader`` kwargs minus ``sampler``/``shuffle``,
            which this function manages. Must include ``batch_size``.
        val_loader_kwargs: same as ``train_loader_kwargs`` for validation.
        lr, eta_min, max_epoch, loss_type: Optimiser/scheduler/strategy params.
        early_stopping_patience: Forwarded to ``TrainerConfig``.
        precision: AMP precision (32/16/bf16).
        log_every_n_steps: Training-metric logging period (in steps).
        static_graph: DDP ``static_graph`` flag.
        checkpoint_base: Root directory for checkpoints; the run subdir is
            ``{checkpoint_base}/{exp_name}/{YYYY-MM-DD-HH:MM-XXXXX}``.
        comet_api_key: Optional Comet API key; empty falls back to env.
        comet_project: Comet project name.
        log_code_file: Optional source file to attach to the experiment.
        metrics_source: Dataset providing ``data_mean_tensor`` / ``data_std_tensor``
            (defaults to ``train_data``). Pass the underlying dataset when
            ``train_data`` is a ``Subset``.
    """
    if train_loader_kwargs is None:
        train_loader_kwargs = {}
    if val_loader_kwargs is None:
        val_loader_kwargs = {}
    if strategy_kwargs is None:
        strategy_kwargs = {}

    dist_info = setup_distributed()
    is_main = dist_info.rank == 0

    train_sampler = (
        DistributedSampler(
            train_data,
            shuffle=train_loader_kwargs.get("shuffle", True),
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
        )
        if dist_info.is_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            valid_data,
            shuffle=False,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
        )
        if dist_info.is_distributed
        else None
    )

    train_loader = DataLoader(
        train_data,
        sampler=train_sampler,
        shuffle=(train_sampler is None) and train_loader_kwargs.get("shuffle", True),
        **{k: v for k, v in train_loader_kwargs.items() if k != "shuffle"},
    )
    val_loader = DataLoader(
        valid_data,
        sampler=val_sampler,
        shuffle=False,
        **{k: v for k, v in val_loader_kwargs.items() if k != "shuffle"},
    )

    steps_per_epoch = len(train_loader) // max(dist_info.world_size, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.9))
    total_steps = (steps_per_epoch + 1) * max_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min
    )

    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls(loss_type=loss_type, **strategy_kwargs)

    mds = metrics_source if metrics_source is not None else train_data
    metrics = Metrics(mds.data_mean_tensor, mds.data_std_tensor)

    checkpoint_dir = os.path.join(checkpoint_base, exp_name, _make_run_id())

    experiment = build_experiment(
        {"comet_project": comet_project, "comet_api_key": comet_api_key},
        experiment_name=exp_name,
        is_main=is_main,
    )
    if log_code_file and is_main:
        experiment.log_code(log_code_file)

    cfg = TrainerConfig(
        max_epochs=max_epoch,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        static_graph=static_graph,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
    )

    if is_main:
        print(f"[experiment] {exp_name}")
        print(f"[checkpoint path] {checkpoint_dir}")
        print(f"[distributed] world_size={dist_info.world_size} rank={dist_info.rank}")

    trainer = Trainer(cfg=cfg, dist_info=dist_info)
    trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=strategy,
        experiment=experiment,
        metrics=metrics,
        full_config={
            "experiment": {"name": exp_name},
            "training": {
                "litmodel": strategy_name,
                "lr": lr,
                "eta_min": eta_min,
                "max_epoch": max_epoch,
                "loss_type": loss_type,
                "early_stopping_patience": early_stopping_patience,
                "extra_kwargs": strategy_kwargs,
            },
            "trainer": {
                "precision": precision,
                "log_every_n_steps": log_every_n_steps,
                "static_graph": static_graph,
            },
            "logging": {
                "checkpoint_base": checkpoint_base,
                "comet_api_key": comet_api_key,
                "comet_project": comet_project,
                "log_code_file": log_code_file,
            },
        },
    )

    experiment.close()
    cleanup_distributed()
