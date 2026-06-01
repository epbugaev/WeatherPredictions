"""PI-IAM4VP training entry point (iterative manual backward), pure-PyTorch."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

PROJECT_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from Data.weatherbench_128_v3 import WeatherBench128
from Models.PI_IAM4VP import IAM4VP
from train._common import run_legacy_training
from utils.regions import USA_CROP


def train_model() -> None:
    """Construct IAM4VP and train with the per-timestep manual-backward strategy."""
    torch_model = IAM4VP()

    cut = USA_CROP
    common = dict(
        include_target=False,
        lead_time=1,
        interval=1,
        muti_target_steps=1,
        start_time_x=0,
        end_time_x=5,
        start_time_y=6,
        end_time_y=11,
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
        exp_name="train_imvp_mini_gft small world physics",
        strategy_name="mutiout_imvp_small_world",
        strategy_kwargs={"time_prediction": 6},
        train_loader_kwargs={"batch_size": 16, "num_workers": 8, "shuffle": True},
        val_loader_kwargs={"batch_size": 16, "num_workers": 8},
        lr=5e-4,
        max_epoch=20,
        log_code_file=os.path.join(PROJECT_ROOT, "Models", "imvp_v2.py"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
