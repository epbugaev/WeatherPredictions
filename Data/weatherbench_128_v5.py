"""v5 — same memmap-backed dataset as v4, registered as a separate version.

v5 itself does NOT change ``__getitem__``: identical raw float32 tensors to
v4. The version bump exists to make the new pipeline (``CudaPrefetcher`` in
``trainer.py`` overlapping CPU→GPU copies with compute via a side CUDA
stream) addressable through the same registry mechanism as v3 / v3_memmap
/ v4.

Activating the prefetcher itself is controlled by ``trainer.cuda_prefetcher:
true`` in the YAML config (not by the dataset class) — the marker just
lets the configs be self-explanatory (``dataset_version: v5`` reads as
"the version designed for the prefetched pipeline").
"""

from __future__ import annotations

from Data.weatherbench_128_v4 import WeatherBench128V4


class WeatherBench128V5(WeatherBench128V4):
    """Identical to ``WeatherBench128V4``; used as the v5 marker class."""
