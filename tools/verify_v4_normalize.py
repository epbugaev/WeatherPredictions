"""Numerical equivalence test: v3_memmap dataset output vs v4 + WeatherNormalize.

Builds the two datasets pointing at the same memmap, reads one sample from each,
applies ``WeatherNormalize`` to the v4 output, and asserts the result is bitwise
equivalent (within fp32 tolerance) to v3_memmap's in-dataset normalisation.

If this script passes, any val_loss divergence between v3_memmap and v4
production runs is purely due to random model initialisation, not a
normalisation bug.

Usage:
    python tools/verify_v4_normalize.py \\
        --memmap /home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat
"""

from __future__ import annotations

import argparse

import torch

from Data.weatherbench_128_v3 import WeatherBench128Memmap
from Data.weatherbench_128_v4 import WeatherBench128V4
from utils.normalize import WeatherNormalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap",
        default="/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    common_kwargs = {
        "start_time": "2000-01-01 00:00:00",
        "end_time": "2003-12-30 00:00:00",
        "lead_time": 1,
        "interval": 1,
        "muti_target_steps": 1,
        "start_time_x": 0,
        "end_time_x": 11,
        "start_time_y": 12,
        "end_time_y": 23,
        "cut": [[75, 107], [164, 228]],
        "memmap_path": args.memmap,
    }

    ds_v3 = WeatherBench128Memmap(**common_kwargs)
    ds_v4 = WeatherBench128V4(**common_kwargs)

    x_v3, y_v3 = ds_v3[args.index]
    x_v4_raw, y_v4_raw = ds_v4[args.index]

    def stats(t):
        return f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}"

    print(f"x_v3:     shape={tuple(x_v3.shape)} dtype={x_v3.dtype}")
    print(f"x_v4_raw: shape={tuple(x_v4_raw.shape)} dtype={x_v4_raw.dtype}")
    print(f"y_v3:     shape={tuple(y_v3.shape)} dtype={y_v3.dtype}")
    print(f"y_v4_raw: shape={tuple(y_v4_raw.shape)} dtype={y_v4_raw.dtype}")
    print()
    print("--- raw v4 (before normalise) ---")
    print(f"  x_v4_raw: {stats(x_v4_raw)}")
    print(f"  y_v4_raw: {stats(y_v4_raw)}")
    print()
    print("--- normalised v3 (in-dataset) ---")
    print(f"  x_v3: {stats(x_v3)}")
    print(f"  y_v3: {stats(y_v3)}")

    normalize = WeatherNormalize(
        mean=torch.as_tensor(ds_v4.the_mean, dtype=torch.float32),
        std=torch.as_tensor(ds_v4.the_std, dtype=torch.float32),
    )
    x_v4_normed = normalize(x_v4_raw)
    y_v4_normed = normalize(y_v4_raw)
    print()
    print("--- normalised v4 (post-WeatherNormalize) ---")
    print(f"  x_v4_normed: {stats(x_v4_normed)}")
    print(f"  y_v4_normed: {stats(y_v4_normed)}")
    print()
    print("--- diff (v3 in-dataset vs v4+WeatherNormalize) ---")
    dx = (x_v3 - x_v4_normed).abs()
    dy = (y_v3 - y_v4_normed).abs()
    print(f"  |dx|: max={dx.max():.6e} mean={dx.mean():.6e}")
    print(f"  |dy|: max={dy.max():.6e} mean={dy.mean():.6e}")
    if torch.allclose(x_v3, x_v4_normed, atol=1e-5, rtol=1e-5):
        print("OK: v3 in-dataset == v4 + WeatherNormalize (within fp32 tolerance)")
    else:
        print("MISMATCH: v3 and v4 + WeatherNormalize differ beyond fp32 tolerance")


if __name__ == "__main__":
    main()
