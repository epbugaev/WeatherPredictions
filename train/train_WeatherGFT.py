"""WeatherGFT training entry point (autoregressive validation, pure-PyTorch)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

PROJECT_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from Data.weatherbench_128 import WeatherBench128
from Models.WeatherGFT import GFT
from train._common import run_legacy_training


def train_model() -> None:
    """Construct WeatherGFT with the original Small World shape and train it."""
    torch_model = GFT(
        hidden_dim=256,
        physics_part_coef=0.1,
        encoder_layers=[2, 2, 2],
        edcoder_heads=[2, 4, 4],
        encoder_scaling_factors=[0.5, 0.5, 1],
        encoder_dim_factors=[-1, 2, 2],
        body_layers=[2, 2, 2, 2, 2, 2],
        body_heads=[6, 6, 6, 6, 6, 6],
        body_scaling_factors=[1, 1, 1, 1, 1, 1],
        body_dim_factors=[1, 1, 1, 1, 1, 1],
        decoder_layers=[2, 2, 2],
        decoder_heads=[4, 4, 2],
        decoder_scaling_factors=[1, 2, 1],
        decoder_dim_factors=[1, 0.5, 1],
        channels=69,
        head_dim=128,
        window_size=[4, 8],
        relative_pos_embedding=False,
        out_kernel=[2, 2],
        pde_block_depth=3,
        block_dt=300,
        inverse_time=False,
        use_checkpoint=True,
    )

    train_data = WeatherBench128(
        start_time="2000-01-01 00:00:00",
        end_time="2003-12-25 00:00:00",
        include_target=False,
        lead_time=1,
        interval=1,
        muti_target_steps=6,
    )
    valid_data = WeatherBench128(
        start_time="2004-01-01 00:00:00",
        end_time="2004-12-25 00:00:00",
        include_target=False,
        lead_time=1,
        interval=1,
        muti_target_steps=12,
    )

    run_legacy_training(
        model=torch_model,
        train_data=train_data,
        valid_data=valid_data,
        exp_name="WeatherGFT classic Small World",
        strategy_name="multiout_double",
        train_loader_kwargs={"batch_size": 16, "num_workers": 4, "shuffle": True},
        val_loader_kwargs={"batch_size": 16, "num_workers": 4},
        lr=1e-4,
        max_epoch=15,
        log_code_file=os.path.join(PROJECT_ROOT, "Models", "WeatherGFTSmallWorld.py"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--nodes", type=int, default=None, help="(legacy, ignored)")
    parser.parse_args()
    train_model()
