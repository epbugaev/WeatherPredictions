"""Сгенерировать синтетический ERA5-like memmap для оффлайн прогона чекеров.

Структура файла совместима с `tools.check_physics_common.open_memmap`:
  * shape: (T, 69, H, W)
  * dtype: float32
  * meta.json: `years`, `hours_per_year`, `shape`, `dtype`.

Каналы 0..68 заполнены физически разумными значениями:
  * 0 t2m   ~ N(285, 10) K
  * 1 u10   ~ N(0, 5) m/s
  * 2 v10   ~ N(0, 5) m/s
  * 3 tp    ~ |N(0, 1e-4)| m
  * 4..16 z @ 13 lvls — линейный профиль hypsometric ~ 5e4 m²/s² (top) → ~1e3 (surface)
  * 17..29 t @ 13 lvls ~ 200 + 60 · (p/1000) K
  * 30..42 r @ 13 lvls ~ U(20, 80) %
  * 43..55 u @ 13 lvls ~ N(0, 15) m/s, + small Rossby-like sinusoid
  * 56..68 v @ 13 lvls ~ N(0, 10) m/s

Это **не** физически согласовано (нет hydrostatic balance), но достаточно для
валидации pipeline: rollout не делится на ноль, метрики считаются, NaN не лезут.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def build_synthetic(
    out_dat: Path,
    year: int = 2005,
    hours: int = 240,
    H: int = 32,
    W: int = 64,
    seed: int = 0,
) -> None:
    """Сгенерировать memmap + meta.json."""
    rng = np.random.default_rng(seed)
    plvls = np.array(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        dtype=np.float32,
    )  # hPa
    arr = np.memmap(out_dat, dtype=np.float32, mode="w+", shape=(hours, 69, H, W))

    # Sinusoidal background: x grid, lat grid
    lat = np.linspace(-np.pi / 2, np.pi / 2, H).reshape(1, H, 1)  # rad
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False).reshape(1, 1, W)

    for t in range(hours):
        # Surface
        arr[t, 0] = 285.0 + 10.0 * rng.standard_normal((H, W)).astype(np.float32)
        arr[t, 1] = 5.0 * rng.standard_normal((H, W)).astype(np.float32)
        arr[t, 2] = 5.0 * rng.standard_normal((H, W)).astype(np.float32)
        arr[t, 3] = np.abs(1e-4 * rng.standard_normal((H, W))).astype(np.float32)

        # Geopotential: log-linear profile 50hPa→5.1e4 m²/s², 1000hPa→8e2.
        # Hypsometric approximation: z(p) ≈ R_d·T·ln(p_ref/p)
        z_profile = 287.0 * 250.0 * np.log(1000.0 / plvls)  # (13,)
        for li, zv in enumerate(z_profile):
            arr[t, 4 + li] = (zv + 1e3 * rng.standard_normal((H, W))).astype(np.float32)

        # Temperature: cooler at top
        t_profile = 200.0 + 0.06 * plvls  # 50hPa→203K, 1000hPa→260K
        for li, tv in enumerate(t_profile):
            arr[t, 17 + li] = (tv + 5.0 * rng.standard_normal((H, W))).astype(np.float32)

        # Relative humidity in %
        for li in range(13):
            arr[t, 30 + li] = (40.0 + 20.0 * rng.standard_normal((H, W))).clip(5, 95).astype(
                np.float32
            )

        # Wind u: zonal, latitude-modulated
        u_profile = 15.0 * np.sin(2 * lat)  # jet ~ at mid-latitudes
        for li in range(13):
            arr[t, 43 + li] = (
                u_profile + 3.0 * np.sin(lon + t * 0.1) + 2.0 * rng.standard_normal((H, W))
            ).astype(np.float32)[0]

        # Wind v
        for li in range(13):
            arr[t, 56 + li] = (
                2.0 * np.cos(2 * lon - t * 0.05) + 1.5 * rng.standard_normal((H, W))
            ).astype(np.float32)[0]

    arr.flush()

    meta = {
        "shape": [hours, 69, H, W],
        "dtype": "float32",
        "years": [year],
        "hours_per_year": [hours],
    }
    meta_path = out_dat.with_suffix(".meta.json")
    if str(out_dat).endswith(".dat"):
        meta_path = Path(str(out_dat)[:-4] + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out_dat} ({arr.nbytes / 1e6:.1f} MB), meta → {meta_path}")


if __name__ == "__main__":
    out = Path("/tmp/synthetic_era5_2005.dat") if len(sys.argv) < 2 else Path(sys.argv[1])
    build_synthetic(out)
