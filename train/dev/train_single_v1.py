"""Experimental: PredFormer (FPredFormer variant) on long-range WeatherBench v2.

The original script imports ``WeatherPredictions.Models.FPredFormer`` which
does not exist in this repository; the import is preserved verbatim to match
the legacy behaviour (the script never ran without that local file).
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from WeatherPredictions.Models.FPredFormer import (
    PredFormer_Model,  # noqa: F401  — legacy local-only import
)

from Data.weatherbench_128_v2 import WeatherBench128
from train._common import run_legacy_training


def train_model() -> None:
    """Construct PredFormer (FPredFormer) and train on a 2000-2016 window."""
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

    common = dict(
        include_target=False,
        lead_time=1,
        interval=1,
        muti_target_steps=1,
        start_time_x=0,
        end_time_x=11,
        start_time_y=12,
        end_time_y=23,
    )
    train_data = WeatherBench128(
        start_time="2000-01-01 00:00:00",
        end_time="2016-12-25 00:00:00",
        **common,
    )
    valid_data = WeatherBench128(
        start_time="2017-01-01 00:00:00",
        end_time="2018-12-25 00:00:00",
        **common,
    )

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="train_predformer_fedor_from_2000_to_2016",
        strategy_name="mutiout_f",
        train_loader_kwargs={"batch_size": 4, "num_workers": 16, "shuffle": True},
        val_loader_kwargs={"batch_size": 4, "num_workers": 16},
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
