"""Environment-driven filesystem defaults for cluster/local runs.

The repo is shared between users, so user-home paths must not be baked into
code. These helpers centralise the few defaults that are still useful on the
HSE cluster while allowing every value to be overridden from ``.env`` or the
submit command.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_WEATHERBENCH_ROOT = "/home/fratnikov/weather_bench"


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def weatherbench_root() -> str:
    """Root containing WeatherBench data directories."""
    return _env("WEATHERBENCH_ROOT", DEFAULT_WEATHERBENCH_ROOT)


def weatherbench_input_root(resolution: str = "1.40625deg") -> str:
    """Directory containing per-variable NetCDF folders for one resolution."""
    return _env("WEATHERBENCH_INPUT_ROOT", str(Path(weatherbench_root()) / resolution))


def weatherbench_npy_root(resolution: str = "1.40625deg") -> str:
    """Directory containing legacy ``YYYY-HHHH.npy`` WeatherBench files."""
    return _env("WEATHERBENCH_NPY_ROOT", str(Path(weatherbench_root()) / "npy" / resolution))


def weatherbench_mean_std_path(resolution: str = "1.40625deg") -> str:
    """Path to legacy ``mean_std.npy`` normalisation statistics."""
    return _env(
        "WEATHERBENCH_MEAN_STD_PATH",
        str(Path(weatherbench_root()) / resolution / "mean_std.npy"),
    )


def weatherbench_constants_path(resolution: str = "1.40625deg") -> str:
    """Path to WeatherBench constants for one resolution."""
    return _env(
        "WEATHERBENCH_CONSTANTS_PATH",
        str(
            Path(weatherbench_input_root(resolution))
            / "constants"
            / f"constants_{resolution}.nc"
        ),
    )


def checkpoint_base() -> str:
    """Base directory for checkpoints."""
    return _env("CHECKPOINT_BASE_OVERRIDE", _env("WEATHERPRED_CHECKPOINT_BASE", "./checkpoints"))


def memmap_dir() -> str:
    """Default directory where packed ERA5 memmaps are stored."""
    return _env("WEATHERPRED_MEMMAP_DIR", str(Path.home() / "era5_memmap"))


def usa_memmap_path() -> str:
    """Default packed USA memmap path."""
    return _env(
        "WEATHERPRED_USA_MEMMAP",
        str(Path(memmap_dir()) / "predformer_usa_2000_2004.dat"),
    )


def globe_memmap_path() -> str:
    """Default packed globe/coarse memmap path used by physics checkers."""
    return _env(
        "WEATHERPRED_GLOBE_MEMMAP",
        str(Path(memmap_dir()) / "predformer_globe_2000_2018.dat"),
    )
