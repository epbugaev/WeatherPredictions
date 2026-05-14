"""PredFormer training entry point (hardcoded config, pure-PyTorch trainer).

A prior version of this script had a broken
``WeatherPredictions.LitModels.mutiout_f`` import (the package never resolved
as written); the strategy registry replaces that path. Launch via ``torchrun``.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.weatherbench_128 import WeatherBench128
from Models.PredFormer import PredFormer_Model
from train._common import run_legacy_training

_CONSTANTS_PATH = "/home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc"


def train_model() -> None:
    """Construct PredFormer + WeatherBench v1 and run training."""
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
        "path_to_constants": _CONSTANTS_PATH,
    }
    torch_model = PredFormer_Model(model_config)

    common = dict(include_target=False, lead_time=1, interval=12, muti_target_steps=23)
    train_data = WeatherBench128(
        start_time="2000-01-01 00:00:00",
        end_time="2003-12-25 00:00:00",
        **common,
    )
    valid_data = WeatherBench128(
        start_time="2004-01-01 00:00:00",
        end_time="2004-12-25 00:00:00",
        **common,
    )

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="train_predformer_raw",
        strategy_name="mutiout_f",
        train_loader_kwargs={"batch_size": 8, "num_workers": 4, "shuffle": True},
        val_loader_kwargs={"batch_size": 8, "num_workers": 4},
        lr=5e-4,
        max_epoch=20,
        checkpoint_base="/home/epbugaev/checkpoints/",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
