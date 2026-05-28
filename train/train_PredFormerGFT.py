"""PredFormerGFT training entry point (USA-style cut, pure-PyTorch trainer)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.weatherbench_128_v3 import WeatherBench128
from Models.PredFormerGFT import PredFormer_Model
from train._common import run_legacy_training
from utils.regions import USA_CROP

_CONSTANTS_PATH = "/home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc"


def train_model() -> None:
    """Construct PredFormerGFT and run training over the regional cut."""
    cut = USA_CROP
    model_config = {
        "height": 32,
        "width": 64,
        "num_channels": 69,
        "pre_seq": 12,
        "after_seq": 12,
        "patch_size": 8,
        "dim": 256,
        "heads": 6,
        "dim_head": 32,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "drop_path": 0.0,
        "scale_dim": 4,
        "depth": 2,
        "Ndepth": 18,
        "path_to_constants": _CONSTANTS_PATH,
        "cut": cut,
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
        cut=cut,
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
        exp_name="train_predformer_gft",
        strategy_name="mutiout_f",
        train_loader_kwargs={"batch_size": 8, "num_workers": 4, "shuffle": True},
        val_loader_kwargs={"batch_size": 8, "num_workers": 4},
        lr=1e-4,
        max_epoch=20,
        checkpoint_base="/home/ebugaev/checkpoints/",
        log_code_file="/home/ebugaev/WeatherPredictions/Models/PredFormerTwoO.py",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
