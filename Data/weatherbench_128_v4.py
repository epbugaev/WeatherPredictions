"""v4 — raw memmap dataset (normalisation lives in the trainer).

Differences from
``Data.weatherbench_128_v3.WeatherBench128Memmap``:
  * ``__getitem__`` returns raw, unnormalised float32 tensors.
  * No ``.float()`` cast (the memmap is already float32; ``.float()`` on a
    float32 tensor is a no-op that adds clutter).
  * No ``np.asarray(...)`` wrap around ``memmap[row]`` (already an
    ``np.ndarray`` view; ``np.asarray`` on an ndarray returns the same
    object).
  * No per-sample arithmetic on the worker — normalisation runs once per
    batch on GPU via ``utils.normalize.WeatherNormalize`` invoked by the
    trainer right after ``_to_device``.

Parent ``__init__`` is reused unchanged: time list, valid_idx, mean/std
loading, memmap mmap, year->row offset map.
"""

from __future__ import annotations

import warnings

import pandas as pd
import torch
from torch.profiler import record_function

from Data.weatherbench_128_v3 import WeatherBench128Memmap

# The memmap is opened read-only, so the np.memmap view's WRITEABLE flag is
# False; torch.from_numpy then emits a UserWarning per worker reminding us
# that writes would be undefined. We never write to these tensors (torch.stack
# below allocates a fresh contiguous buffer, and WeatherNormalize returns a
# new tensor) — so the warning is pure noise. Silence it once on import.
warnings.filterwarnings(
    "ignore",
    message=r".*NumPy array is not writable.*",
    category=UserWarning,
)


class WeatherBench128V4(WeatherBench128Memmap):
    """Raw-memmap dataset; normalisation handled by the model pipeline.

    Each sample is a pair of contiguous float32 tensors:
        x: shape ``(T_input, C, H, W)``
        y: shape ``(T_target, C, H, W)`` if ``muti_target_steps == 1``
           or  ``(S, T_target, C, H, W)`` otherwise.

    Overrides only ``_read_x``/``_read_y``: skips per-sample normalisation
    (the trainer's ``WeatherNormalize`` runs once per batch on GPU) and the
    redundant ``.float()`` cast (the memmap is already float32).
    """

    def _read_x(self, time_idx: int) -> torch.Tensor:
        t = self.x_time_ilst[time_idx]
        with record_function("memmap_read_x"):
            # self._memmap[row]: np.memmap view, shape (C, H, W), float32
            row = self._memmap[self._memmap_row(t)]
        with record_function("from_numpy_x"):
            # torch.from_numpy shares memory with the np.memmap view;
            # the actual page-in happens at torch.stack in the base loop.
            return torch.from_numpy(row)

    def _read_y(self, y_time: pd.Timestamp) -> torch.Tensor:
        with record_function("memmap_read_y"):
            row = self._memmap[self._memmap_row(y_time)]
        with record_function("from_numpy_y"):
            return torch.from_numpy(row)
