"""Experimental: IAM4VP (imvp_v1) on the OpenSTL-style WeatherBenchDataset2."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Data.dataloader_weather import WeatherBenchDataset2
from Models.dev.imvp_v1 import IAM4VP
from train._common import run_legacy_training


def _prepare_mean_or_std(data) -> torch.Tensor:
    """Tile the OpenSTL per-level stats out to the 69-channel layout."""
    data = torch.from_numpy(data).squeeze()
    z_old, t_old, q_old, u_old, v_old = data[4:].chunk(5, dim=0)
    z_old = z_old.repeat(13)
    t_old = t_old.repeat(13)
    q_old = q_old.repeat(13)
    u_old = u_old.repeat(13)
    v_old = v_old.repeat(13)
    return torch.cat([data[:4], z_old, t_old, q_old, u_old, v_old], dim=0)


class _MetricsAdapter(Dataset):
    """Thin shim that exposes ``data_mean_tensor`` / ``data_std_tensor`` on top of ``ds``."""

    def __init__(self, ds: Dataset, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._ds = ds
        self.data_mean_tensor = mean
        self.data_std_tensor = std

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]


def train_model() -> None:
    """Train IAM4VP-v1 on WeatherBenchDataset2 (OpenSTL data root)."""
    torch_model = IAM4VP()

    raw_train = WeatherBenchDataset2(
        data_root="/home/fratnikov/weather_bench/1.40625deg",
        data_name="gft",
        training_time=["2013", "2015"],
        idx_in=list(range(0, 6)),
        idx_out=list(range(6, 12)),
        step=1,
        levels="all",
        data_split="1_40625",
        use_augment=False,
    )
    raw_valid = WeatherBenchDataset2(
        data_root="/home/fratnikov/weather_bench/1.40625deg",
        data_name="gft",
        training_time=["2016", "2016"],
        idx_in=list(range(0, 6)),
        idx_out=list(range(6, 12)),
        step=1,
        levels="all",
        data_split="1_40625",
        use_augment=False,
    )

    mean = _prepare_mean_or_std(raw_train.mean)
    std = _prepare_mean_or_std(raw_train.std)
    train_data = _MetricsAdapter(raw_train, mean=mean, std=std)
    valid_data = _MetricsAdapter(raw_valid, mean=mean, std=std)

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="train_imvp_mini_gft_v2",
        strategy_name="mutiout_imvp",
        strategy_kwargs={"time_prediction": 6},
        train_loader_kwargs={"batch_size": 16, "num_workers": 32, "shuffle": True},
        val_loader_kwargs={"batch_size": 16, "num_workers": 32},
        lr=5e-4,
        max_epoch=20,
        checkpoint_base="/home/fa.buzaev/checkpoints/",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
