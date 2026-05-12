"""SimVP training entry point (hardcoded config, pure-PyTorch trainer).

Delegates the loop to ``train._common.run_legacy_training``. Launch via
``torchrun`` (see ``sh_files/launch_train.sh``).
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.weatherbench_128_v3 import WeatherBench128
from Models.SimVP import SimVP_Model
from train._common import run_legacy_training


def train_model() -> None:
    """Build hardcoded model + datasets and hand them to the trainer."""
    torch_model = SimVP_Model(in_shape=(12, 69, 32, 64))

    cut = [[128 - 92, 128 - 60], [256 - 131, 256 - 67]]
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
        end_time="2003-12-30 00:00:00",
        **common,
    )
    valid_data = WeatherBench128(
        start_time="2004-01-01 00:00:00",
        end_time="2004-12-30 00:00:00",
        **common,
    )

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="SimVP",
        strategy_name="mutiout_f",
        strategy_kwargs={"time_prediction": 6},
        train_loader_kwargs={"batch_size": 32, "num_workers": 8, "shuffle": True},
        val_loader_kwargs={"batch_size": 32, "num_workers": 8},
        lr=5e-4,
        max_epoch=20,
        log_code_file="/home/ebugaev/WeatherPredictions/Models/SimVP.py",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
