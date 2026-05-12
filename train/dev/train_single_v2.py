"""Experimental: PredFormer (FPredFormer) on the OpenSTL WeatherBenchDataset2.

The ``WeatherPredictions.Models.FPredFormer`` import is missing from the
repository and is preserved verbatim from the legacy script.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from WeatherPredictions.Models.FPredFormer import (
    PredFormer_Model,  # noqa: F401  — legacy local-only import
)

from Data.dataloader_weather import WeatherBenchDataset2
from train._common import run_legacy_training


def train_model() -> None:
    """Construct PredFormer (FPredFormer) and train on the 2015/2016 split."""
    model_config = {
        "height": 128,
        "width": 256,
        "num_channels": 69,
        "pre_seq": 12,
        "after_seq": 12,
        "patch_size": 8,
        "dim": 256,
        "heads": 8,
        "dim_head": 32,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "drop_path": 0.0,
        "scale_dim": 4,
        "depth": 1,
        "Ndepth": 24,
        "path_to_constants": (
            "/home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc"
        ),
    }
    torch_model = PredFormer_Model(model_config)

    train_data = WeatherBenchDataset2(
        data_root="/home/fa.buzaev/data/weatherbench",
        data_name="gft",
        training_time=["2015", "2015"],
        idx_in=list(range(0, 12)),
        idx_out=list(range(12, 24)),
        step=12,
        levels="all",
        data_split="1_40625",
        use_augment=False,
    )
    valid_data = WeatherBenchDataset2(
        data_root="/home/fa.buzaev/data/weatherbench",
        data_name="gft",
        training_time=["2016", "2016"],
        idx_in=list(range(0, 12)),
        idx_out=list(range(12, 24)),
        step=12,
        levels="all",
        data_split="1_40625",
        use_augment=False,
    )

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="train_predformer_fedor_from_2015_to_2016",
        strategy_name="mutiout_f",
        train_loader_kwargs={"batch_size": 4, "num_workers": 8, "shuffle": True},
        val_loader_kwargs={"batch_size": 4, "num_workers": 8},
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
