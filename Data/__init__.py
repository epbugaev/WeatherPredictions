"""Dataset package: registers WeatherBench128 versions under their YAML keys.

Each version (``v1``, ``v2``, ``v3``) has slightly different ``__init__``
signatures; the registered factory accepts the common ``params`` dict from
``train.py`` and forwards only the keys the underlying class expects.
"""

from Data.weatherbench_128 import WeatherBench128 as WeatherBench128V1
from Data.weatherbench_128_v2 import WeatherBench128 as WeatherBench128V2
from Data.weatherbench_128_v3 import (
    WeatherBench128 as WeatherBench128V3,
)
from Data.weatherbench_128_v3 import (
    WeatherBench128Memmap as WeatherBench128V3Memmap,
)
from Data.weatherbench_128_v4 import WeatherBench128V4
from utils.registry import register_dataset


def _build_v1(**params):
    """v1 ignores the explicit start/end_time_x/y parameters (legacy)."""
    for legacy_key in ("start_time_x", "end_time_x", "start_time_y", "end_time_y"):
        params.pop(legacy_key, None)
    return WeatherBench128V1(**params)


def _build_v2(**params):
    """v2 ignores the explicit start/end_time_x/y parameters."""
    for legacy_key in ("start_time_x", "end_time_x", "start_time_y", "end_time_y"):
        params.pop(legacy_key, None)
    return WeatherBench128V2(**params)


def _build_v3(**params):
    """v3 uses start/end_time_x/y to size input and target sequences."""
    return WeatherBench128V3(**params)


def _build_v3_memmap(**params):
    """v3_memmap reads from a pre-packed np.memmap; see ``tools/repack_era5.py``."""
    return WeatherBench128V3Memmap(**params)


def _build_v4(**params):
    """v4: raw memmap, normalisation lives in the trainer (WeatherNormalize)."""
    return WeatherBench128V4(**params)


register_dataset("v1")(_build_v1)
register_dataset("v2")(_build_v2)
register_dataset("v3")(_build_v3)
register_dataset("v3_memmap")(_build_v3_memmap)
register_dataset("v4")(_build_v4)
