"""Unified training entry point driven by YAML config files.

Run with ``torchrun`` for single- or multi-GPU/multi-node training::

    torchrun --standalone --nproc_per_node=1 train.py --config configs/simvp.yaml
    torchrun --nnodes=2 --nproc_per_node=8 train.py --config configs/simvp.yaml

``train.py`` reads the YAML, looks up the model / dataset / strategy by their
registry keys (no local imports, no model-specific if-chains) and hands
everything to ``trainer.Trainer.fit``. DDP, AMP, checkpointing, early stopping
and Comet logging all live in dedicated ``utils`` modules.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import re
import string
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

import Data  # noqa: F401  — registers datasets in utils.registry.DATASETS
import Models  # noqa: F401  — registers models in utils.registry.MODELS
import training_strategies  # noqa: F401  — registers strategies in utils.registry.STRATEGIES
from trainer import Trainer, TrainerConfig
from utils.distributed import cleanup_distributed, setup_distributed
from utils.experiment import build_experiment
from utils.metrics import Metrics
from utils.normalize import WeatherNormalize
from utils.paths import checkpoint_base as default_checkpoint_base
from utils.registry import get_dataset, get_model, get_strategy


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_value(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in YAML values."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    with open(path) as f:
        return _expand_env_value(yaml.safe_load(f))


def build_dataset(data_cfg: dict[str, Any], split: str):
    """Build the dataset for ``split`` from the YAML ``data`` section.

    Args:
        data_cfg: ``data`` block from the YAML config.
        split: ``"train"`` or ``"val"``; split-level keys override common ones.

    Returns:
        A ``torch.utils.data.Dataset`` instance.
    """
    version = data_cfg.get("dataset_version", "v3")
    split_cfg = data_cfg.get(split, {})

    params: dict[str, Any] = {
        "start_time": split_cfg["start_time"],
        "end_time": split_cfg["end_time"],
        "include_target": split_cfg.get("include_target", data_cfg.get("include_target", False)),
        "lead_time": split_cfg.get("lead_time", data_cfg.get("lead_time", 1)),
        "interval": split_cfg.get("interval", data_cfg.get("interval", 1)),
        "muti_target_steps": split_cfg.get(
            "muti_target_steps", data_cfg.get("muti_target_steps", 1)
        ),
        "start_time_x": data_cfg.get("start_time_x", 0),
        "end_time_x": data_cfg.get("end_time_x", 1),
        "start_time_y": data_cfg.get("start_time_y", 0),
        "end_time_y": data_cfg.get("end_time_y", 1),
    }
    for key in ("sample_stride", "frame_interval"):
        value = split_cfg.get(key, data_cfg.get(key))
        if value is not None:
            params[key] = value

    cut = split_cfg.get("cut", data_cfg.get("cut"))
    if cut is not None:
        params["cut"] = cut

    # Optional storage-path overrides. Forwarded only when set, so the
    # underlying dataset class doesn't see kwargs it doesn't accept.
    # Environment variables (``DATA_FOLDER_OVERRIDE`` etc.) take precedence
    # over the YAML value — used by submit scripts that stage data to local
    # NVMe and need to redirect Dataset reads to the staged copy without
    # rewriting the committed YAML.
    for key in (
        "data_folder",
        "input_folder",
        "mean_std_path",
        "memmap_path",
        "memmap_meta_path",
    ):
        env_key = key.upper() + "_OVERRIDE"
        env_value = os.environ.get(env_key)
        if env_value:
            params[key] = env_value
            continue
        if key == "data_folder":
            env_value = os.environ.get("WEATHERBENCH_NPY_ROOT")
        elif key == "input_folder":
            env_value = os.environ.get("WEATHERBENCH_INPUT_ROOT")
        elif key == "mean_std_path":
            env_value = os.environ.get("WEATHERBENCH_MEAN_STD_PATH")
        if env_value:
            params[key] = env_value
            continue
        value = data_cfg.get(key)
        if value is not None:
            params[key] = value

    return get_dataset(version)(**params)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_cfg: dict[str, Any],
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build AdamW with optional LR warmup followed by CosineAnnealingLR.

    The original LitModels used ``betas=(0.9, 0.9)`` and ``weight_decay=0.0``;
    ``betas`` are preserved for numerical parity. ``weight_decay`` defaults to
    ``0.0`` (parity) but is now read from ``training.weight_decay`` so it can
    be enabled as an L2 regulariser against the observed overfit. The
    scheduler's ``T_max`` is ``(steps_per_epoch + 1) * max_epoch`` exactly as
    in ``LitModels/basemodel.py``.

    When ``training.warmup_ratio`` > 0, the first ``warmup_ratio * total_steps``
    steps run a linear warmup from ``warmup_start_factor * lr`` up to ``lr``;
    the cosine schedule then anneals over the remaining steps. Used at larger
    batch sizes where a cold ``lr`` is too aggressive in the first few epochs
    and the loss "jumps off" the basin found in epoch 1.
    """
    lr = float(training_cfg.get("lr", 1e-4))
    eta_min = float(training_cfg.get("eta_min", 0.0))
    max_epoch = int(training_cfg.get("max_epoch", 20))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
    warmup_start_factor = float(training_cfg.get("warmup_start_factor", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.9),
        fused=torch.cuda.is_available(),
    )
    total_steps = (steps_per_epoch + 1) * max_epoch
    if warmup_ratio > 0.0:
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )
    return optimizer, scheduler


def build_strategy(training_cfg: dict[str, Any]):
    """Instantiate the ``StepStrategy`` indicated by ``training.litmodel``."""
    litmodel_type = training_cfg.get("litmodel", "mutiout_f")
    extra = dict(training_cfg.get("extra_kwargs", {}))
    extra["loss_type"] = training_cfg.get("loss_type", "MAE")
    strategy_cls = get_strategy(litmodel_type)
    return strategy_cls(**extra)


def _make_run_id() -> str:
    """Compose the run-id used as the checkpoint subdir name."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + "".join(
        random.choices(string.ascii_lowercase + string.digits, k=5)
    )


def train(config: dict[str, Any], config_path: str | None = None) -> None:
    """End-to-end training routine driven entirely by ``config``."""
    dist_info = setup_distributed()
    is_main = dist_info.rank == 0

    # Optional reproducibility: set ``training.seed`` in the YAML config to a
    # non-negative int to fix model init / DistributedSampler / DataLoader
    # worker seeds. Used for A/B-style numerical comparisons (e.g. v3_memmap
    # vs v4 with WeatherNormalize). Default unset — non-deterministic init.
    seed = config.get("training", {}).get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    model = get_model(config["model"]["type"])(**config["model"].get("params", {}))
    model = model.to(dist_info.device)

    data_cfg = config["data"]
    train_data = build_dataset(data_cfg, "train")
    valid_data = build_dataset(data_cfg, "val")

    train_subset_step = data_cfg.get("train", {}).get("subset_step")
    if train_subset_step:
        indices = list(range(0, len(train_data), train_subset_step))[:-5]
        train_data = Subset(train_data, indices)

    train_cfg = data_cfg.get("train", {})
    val_cfg = data_cfg.get("val", {})
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", torch.cuda.is_available())
    persistent_workers = data_cfg.get("persistent_workers", num_workers > 0)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_data,
            shuffle=train_cfg.get("shuffle", True),
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
        )
        if dist_info.is_distributed
        else None
    )
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            valid_data,
            shuffle=False,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
        )
        if dist_info.is_distributed
        else None
    )

    # ``persistent_workers`` and ``prefetch_factor`` are only valid when
    # ``num_workers > 0``; passing them with 0 workers raises in PyTorch.
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=(train_sampler is None) and train_cfg.get("shuffle", True),
        sampler=train_sampler,
        drop_last=train_cfg.get("drop_last", True),
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=val_cfg.get("batch_size", 16),
        shuffle=False,
        sampler=val_sampler,
        drop_last=val_cfg.get("drop_last", False),
        **loader_kwargs,
    )

    training_cfg = config["training"]
    # With DistributedSampler, DataLoader length is already the per-rank number
    # of optimizer steps. Dividing by world_size again would compress warmup and
    # cosine schedules on DDP runs.
    steps_per_epoch = len(train_loader)

    metrics_source = train_data.dataset if isinstance(train_data, Subset) else train_data
    metrics = Metrics(metrics_source.data_mean_tensor, metrics_source.data_std_tensor)

    # v4 returns raw memmap rows; legacy v1/v3/v3_memmap datasets already
    # normalize inside __getitem__. Keep normalization in exactly one place.
    normalize = None
    if not getattr(metrics_source, "returns_normalized", True):
        normalize = WeatherNormalize(
            mean=torch.as_tensor(metrics_source.the_mean, dtype=torch.float32),
            std=torch.as_tensor(metrics_source.the_std, dtype=torch.float32),
        )
    set_physics_normalization = getattr(model, "set_physics_normalization", None)
    if set_physics_normalization is not None:
        set_physics_normalization(metrics_source.data_mean_tensor, metrics_source.data_std_tensor)

    optimizer, scheduler = build_optimizer_and_scheduler(model, training_cfg, steps_per_epoch)
    strategy = build_strategy(training_cfg)

    exp_cfg = config.get("experiment", {})
    logging_cfg = config.get("logging", {})
    exp_name = exp_cfg.get("name", "experiment")
    checkpoint_root = os.environ.get(
        "CHECKPOINT_BASE_OVERRIDE",
        logging_cfg.get("checkpoint_base") or default_checkpoint_base(),
    )
    checkpoint_dir = os.path.join(checkpoint_root, exp_name, _make_run_id())

    experiment = build_experiment(logging_cfg, experiment_name=exp_name, is_main=is_main)
    log_code_file = logging_cfg.get("log_code_file")
    if log_code_file and is_main:
        experiment.log_code(log_code_file)

    trainer_cfg = config.get("trainer", {})
    cfg = TrainerConfig(
        max_epochs=training_cfg.get("max_epoch", 20),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 5),
        static_graph=trainer_cfg.get("static_graph", True),
        find_unused_parameters=trainer_cfg.get("find_unused_parameters", False),
        early_stopping_patience=training_cfg.get("early_stopping_patience", 5),
        checkpoint_dir=checkpoint_dir,
        float32_matmul_precision=trainer_cfg.get("float32_matmul_precision"),
        grad_clip_norm=trainer_cfg.get("grad_clip_norm"),
        skip_non_finite_loss=trainer_cfg.get("skip_non_finite_loss", True),
        val_every_n_epochs=trainer_cfg.get("val_every_n_epochs", 1),
    )

    if is_main:
        print(f"[config] {os.path.abspath(config_path) if config_path else 'N/A'}")
        print(f"[experiment] {exp_name}")
        print(f"[checkpoint path] {checkpoint_dir}")
        print(f"[distributed] world_size={dist_info.world_size} rank={dist_info.rank}")

    trainer = Trainer(cfg=cfg, dist_info=dist_info)
    trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=strategy,
        experiment=experiment,
        metrics=metrics,
        full_config=config,
        normalize=normalize,
    )

    experiment.close()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, config_path=args.config)
