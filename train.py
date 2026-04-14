"""Unified training script driven by YAML config files.

Usage:
    python train.py --config configs/simvp.yaml
    python train.py --config configs/simvp.yaml --gpus_per_node 8 --nodes 1
"""

import comet_ml

import yaml
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser
import datetime
import os
import random
import string

from utils.metrics import Metrics
from lightning.pytorch.loggers import CometLogger


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    """Instantiate a model from the config's model section."""
    model_type = model_cfg["type"]
    params = model_cfg.get("params", {})

    if model_type == "SimVP":
        from Models.SimVP import SimVP_Model
        if "in_shape" in params:
            params["in_shape"] = tuple(params["in_shape"])
        return SimVP_Model(**params)

    elif model_type == "WeatherGFT":
        from Models.WeatherGFT import GFT
        return GFT(**params)

    elif model_type == "WeatherGFTSingle":
        from Models.WeatherGFTSingle import GFT
        return GFT(**params)

    elif model_type == "PredFormerGFT":
        from Models.PredFormerGFT import PredFormer_Model
        return PredFormer_Model(params)

    elif model_type == "PredFormer":
        from Models.PredFormer import PredFormer_Model
        return PredFormer_Model(params)

    elif model_type == "PI-IAM4VP":
        from Models.PI_IAM4VP import IAM4VP
        return IAM4VP(**params)

    elif model_type == "PredRNN":
        from Models.PredRNN import PredRNN_Model
        configs = params.pop("configs", {})
        if "in_shape" in configs:
            configs["in_shape"] = tuple(configs["in_shape"])
        if "num_hidden" in params:
            params["num_hidden"] = tuple(params["num_hidden"])
        return PredRNN_Model(configs=configs, **params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_dataset(data_cfg, split):
    """Build a dataset for the given split ('train' or 'val').

    Parameters from the split sub-key override common data-level parameters.
    """
    version = data_cfg.get("dataset_version", "v3")
    split_cfg = data_cfg.get(split, {})

    params = {
        "start_time": split_cfg["start_time"],
        "end_time": split_cfg["end_time"],
        "include_target": split_cfg.get("include_target", data_cfg.get("include_target", False)),
        "lead_time": split_cfg.get("lead_time", data_cfg.get("lead_time", 1)),
        "interval": split_cfg.get("interval", data_cfg.get("interval", 1)),
        "muti_target_steps": split_cfg.get("muti_target_steps", data_cfg.get("muti_target_steps", 1)),
    }

    cut = split_cfg.get("cut", data_cfg.get("cut"))
    if cut is not None:
        params["cut"] = cut

    if version == "v3":
        from Data.weatherbench_128_v3 import WeatherBench128
        params["start_time_x"] = data_cfg.get("start_time_x", 0)
        params["end_time_x"] = data_cfg.get("end_time_x", 1)
        params["start_time_y"] = data_cfg.get("start_time_y", 0)
        params["end_time_y"] = data_cfg.get("end_time_y", 1)
    elif version == "v2":
        from Data.weatherbench_128_v2 import WeatherBench128
    elif version == "v1":
        from Data.weatherbench_128 import WeatherBench128
    else:
        raise ValueError(f"Unknown dataset version: {version}")

    return WeatherBench128(**params)


def build_litmodel(training_cfg, model, metrics, steps_per_epoch):
    """Wrap the model in a LightningModule training wrapper."""
    litmodel_type = training_cfg.get("litmodel", "mutiout_f")

    kwargs = {
        "model": model,
        "lr": training_cfg.get("lr", 1e-4),
        "eta_min": training_cfg.get("eta_min", 0.0),
        "max_epoch": training_cfg.get("max_epoch", 20),
        "steps_per_epoch": steps_per_epoch,
        "loss_type": training_cfg.get("loss_type", "MAE"),
        "metrics": metrics,
    }

    extra = training_cfg.get("extra_kwargs", {})
    kwargs.update(extra)

    if litmodel_type == "mutiout_f":
        from LitModels.mutiout_f import MutiOut
    elif litmodel_type == "multiout_double":
        from LitModels.multiout_double import MutiOut
    elif litmodel_type == "mutiout":
        from LitModels.mutiout import MutiOut
    elif litmodel_type == "mutiout_imvp":
        from LitModels.mutiout_imvp import MutiOut
    elif litmodel_type == "mutiout_imvp_small_world":
        from LitModels.mutiout_imvp_small_world import MutiOut
    elif litmodel_type == "mutiout_predrnn":
        from LitModels.mutiout_predrnn import MutiOut
    else:
        raise ValueError(f"Unknown litmodel type: {litmodel_type}")

    return MutiOut(**kwargs)


def train(config, devices, num_nodes, config_path=None):
    # ── Model ──
    torch_model = build_model(config["model"])

    # ── Data ──
    data_cfg = config["data"]
    train_data = build_dataset(data_cfg, "train")
    valid_data = build_dataset(data_cfg, "val")

    # Optional subset sampling (e.g. PredRNN takes every Nth sample)
    train_subset_step = data_cfg.get("train", {}).get("subset_step")
    if train_subset_step:
        indices = list(range(0, len(train_data), train_subset_step))[:-5]
        train_data_raw = train_data
        train_data = Subset(train_data, indices)

    train_cfg = data_cfg.get("train", {})
    val_cfg = data_cfg.get("val", {})
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=train_cfg.get("shuffle", True),
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=val_cfg.get("batch_size", 16),
        shuffle=val_cfg.get("shuffle", False),
        num_workers=num_workers,
    )

    # ── LitModel ──
    training_cfg = config["training"]
    world_size = devices * num_nodes
    steps_per_epoch = len(train_loader) // world_size

    # Metrics need mean/std tensors — handle Subset wrapper
    metrics_source = train_data.dataset if isinstance(train_data, Subset) else train_data
    metrics = Metrics(metrics_source.data_mean_tensor, metrics_source.data_std_tensor)

    lit_model = build_litmodel(training_cfg, torch_model, metrics, steps_per_epoch)

    # ── Checkpoint & callbacks ──
    exp_cfg = config.get("experiment", {})
    logging_cfg = config.get("logging", {})

    exp_name = exp_cfg.get("name", "experiment")
    checkpoint_base = logging_cfg.get("checkpoint_base", "./checkpoints/")
    run_id = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    )
    save_path = os.path.join(checkpoint_base, exp_name, run_id)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training_cfg.get("early_stopping_patience", 5),
        check_finite=True,
    )

    # ── Comet ML ──
    comet_api_key = logging_cfg.get("comet_api_key", "")
    os.environ["COMET_API_KEY"] = comet_api_key
    os.environ["COMET_EXPERIMENT_KEY"] = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=50)
    )
    comet_ml.login()

    comet_project = logging_cfg.get("comet_project", "WeatherPredictions")
    logger = CometLogger(project_name=comet_project, experiment_name=exp_name)

    log_code_file = logging_cfg.get("log_code_file")
    if log_code_file:
        logger.experiment.log_code(file_name=log_code_file)

    # ── Trainer ──
    trainer_cfg = config.get("trainer", {})
    trainer = L.Trainer(
        default_root_dir="./",
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 5),
        precision=trainer_cfg.get("precision", 32),
        max_epochs=training_cfg.get("max_epoch", 20),
        logger=logger,
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=DDPStrategy(static_graph=trainer_cfg.get("static_graph", True)),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.print(f"[config] {os.path.abspath(config_path) if config_path else 'N/A'}")
    trainer.print(f"[experiment] {exp_name}")
    trainer.print(f"[checkpoint path] {save_path}")

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    trainer.print("train over")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, devices=args.gpus_per_node, num_nodes=args.nodes, config_path=args.config)
