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

import pandas as pd
import torch
from torch.profiler import record_function

from Data.weatherbench_128_v3 import WeatherBench128Memmap


class WeatherBench128V4(WeatherBench128Memmap):
    """Raw-memmap dataset; normalisation handled by the model pipeline.

    Each sample is a pair of contiguous float32 tensors:
        x: shape ``(T_input, C, H, W)``
        y: shape ``(T_target, C, H, W)`` if ``muti_target_steps == 1``
           or  ``(S, T_target, C, H, W)`` otherwise.
    """

    def __getitem__(self, index):
        x_sequence = []
        for i in range(self.start_time_x, self.end_time_x + 1):
            t = self.x_time_ilst[index + i]
            with record_function("memmap_read_x"):
                # self._memmap[row]: np.memmap view, shape (C, H, W), float32
                row = self._memmap[self._memmap_row(t)]
            with record_function("from_numpy_x"):
                # torch.from_numpy shares memory with the np.memmap view;
                # the actual page-in happens at torch.stack below.
                x_sequence.append(torch.from_numpy(row))

        with record_function("stack_x"):
            # (T_input, C, H, W), torch.float32, contiguous.
            sample_x_sequence = torch.stack(x_sequence, dim=0)

        y_sequences = []
        for steps in range(self.muti_target_steps):
            y_sequence = []
            for i in range(self.start_time_y, self.end_time_y + 1):
                x_time = self.x_time_ilst[index + i]
                y_time = x_time + pd.Timedelta(hours=(steps + 1) * self.lead_time)
                with record_function("memmap_read_y"):
                    row = self._memmap[self._memmap_row(y_time)]
                with record_function("from_numpy_y"):
                    y_sequence.append(torch.from_numpy(row))

            with record_function("stack_y"):
                y_sequences.append(torch.stack(y_sequence, dim=0))

        if self.muti_target_steps > 1:
            # (S, T_target, C, H, W), torch.float32.
            sample_y_all = torch.stack(y_sequences, dim=0)
        else:
            # (T_target, C, H, W), torch.float32.
            sample_y_all = y_sequences[0]
        return sample_x_sequence, sample_y_all
