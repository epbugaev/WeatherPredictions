"""PredRNN training entry point with Subset sampling, pure-PyTorch trainer."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

from torch.utils.data import Subset

PROJECT_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from Models.PredRNN import PredRNN_Model

from Data.weatherbench_128_v3 import WeatherBench128
from train._common import run_legacy_training
from utils.regions import USA_CROP


def train_model() -> None:
    """Construct PredRNN, sub-sample the training data, and train."""
    config = {
        "in_shape": (12, 69, 32, 64),
        "patch_size": 1,
        "filter_size": 3,
        "stride": 1,
        "layer_norm": True,
        "pre_seq_length": 12,
        "aft_seq_length": 12,
        "reverse_scheduled_sampling": 0,
    }
    torch_model = PredRNN_Model(num_layers=4, num_hidden=(128, 128, 128, 128), configs=config)

    cut = USA_CROP
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
    train_full = WeatherBench128(
        start_time="2000-01-01 00:00:00",
        end_time="2003-12-30 00:00:00",
        **common,
    )
    valid_data = WeatherBench128(
        start_time="2004-01-01 00:00:00",
        end_time="2004-01-25 00:00:00",
        **common,
    )

    indices = list(range(0, len(train_full), 23))[:-5]
    train_data = Subset(train_full, indices)

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="PredRNN",
        strategy_name="mutiout_predrnn",
        strategy_kwargs={"time_prediction": 6},
        train_loader_kwargs={"batch_size": 8, "num_workers": 8, "shuffle": True},
        val_loader_kwargs={"batch_size": 8, "num_workers": 8},
        lr=1e-4,
        max_epoch=4,
        metrics_source=valid_data,
        log_code_file=os.path.join(PROJECT_ROOT, "Models", "imvp_v2.py"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
