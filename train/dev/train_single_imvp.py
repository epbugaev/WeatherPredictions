"""Experimental: IAM4VP (imvp_v2) trained with the iterative-manual strategy."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Data.weatherbench_128_v3 import WeatherBench128
from Models.dev.imvp_v2 import IAM4VP
from train._common import run_legacy_training


def train_model() -> None:
    """Construct IAM4VP from Models/dev and train it on WeatherBench v3."""
    torch_model = IAM4VP()

    common = dict(
        include_target=False,
        lead_time=1,
        interval=1,
        muti_target_steps=1,
        start_time_x=0,
        end_time_x=5,
        start_time_y=6,
        end_time_y=11,
    )
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
        exp_name="f train_imvp_mini_gft",
        strategy_name="mutiout_imvp",
        strategy_kwargs={"time_prediction": 6},
        train_loader_kwargs={"batch_size": 8, "num_workers": 8, "shuffle": True},
        val_loader_kwargs={"batch_size": 8, "num_workers": 8},
        lr=5e-4,
        max_epoch=20,
        log_code_file="/home/ebugaev/WeatherPredictions/Models/imvp_v2.py",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
