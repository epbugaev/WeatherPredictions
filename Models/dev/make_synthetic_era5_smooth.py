"""Гладкий, ~сбалансированный синтетический ERA5-memmap для ЛОКАЛЬНОГО
re-run абляции (когда кластер недоступен).

В отличие от `make_synthetic_era5.py` (бело-шумный — взрывается у ВСЕХ
методов мгновенно, не различает их), здесь поля НИЗКО-ВОЛНОВЫЕ и
~гидростатически/геострофически согласованы:
  * z(p) = R_d·T̄·ln(p_s/p) + крупномасштабная волна A·cosφ·cos(2λ)
  * T = T̄ + малая гладкая поправка
  * (u,v) — геострофические из z (с тейпером у экватора, |f|→0)
  * q,r,surface — гладкие, малые
→ чистая физика РАЗЛИЧАЕТ методы: baseline (E0) взрывается за ~12
substep'ов (CFL/алиасинг), стабилизаторы (E2/E3/E6) продлевают
устойчивость — это и есть локальная проверка «проведи ещё раз опыты».

Структура файла — как у make_synthetic_era5 (T,69,H,W) + .meta.json.

Запуск: .venv/bin/python Models/dev/make_synthetic_era5_smooth.py /tmp/syn_smooth.dat
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R_D = 287.0
OMEGA = 7.2921e-5


def build_smooth(
    out_dat: Path, year: int = 2005, hours: int = 240, H: int = 32, W: int = 64
) -> None:
    """Гладкий сбалансированный memmap (детерминирован, без шума)."""
    plvls = np.array(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.float64
    )
    arr = np.memmap(out_dat, dtype=np.float32, mode="w+", shape=(hours, 69, H, W))

    lat = np.linspace(-np.pi / 2, np.pi / 2, H + 2)[1:-1].reshape(H, 1)  # без полюсов
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False).reshape(1, W)
    cphi = np.cos(lat)
    f = 2.0 * OMEGA * np.sin(lat)
    f_safe = np.where(
        np.abs(f) < 2.0 * OMEGA * np.sin(np.deg2rad(5.0)),
        np.sign(f + 1e-30) * 2.0 * OMEGA * np.sin(np.deg2rad(5.0)),
        f,
    )
    radius = 6.371e6
    t_bar = 250.0

    for t in range(hours):
        # медленно вращающаяся крупномасштабная волна (k=2 по долготе)
        phase = 0.04 * t
        wave = (cphi * np.cos(2.0 * lon + phase)).astype(np.float64)  # (H,W), O(1)
        dwave_dlon = cphi * -2.0 * np.sin(2.0 * lon + phase)
        dwave_dlat = -np.sin(lat) * np.cos(2.0 * lon + phase)

        arr[t, 0] = (285.0 + 5.0 * wave).astype(np.float32)  # t2m
        arr[t, 1] = (3.0 * wave).astype(np.float32)  # u10
        arr[t, 2] = (2.0 * wave).astype(np.float32)  # v10
        arr[t, 3] = np.abs(1e-4 * wave).astype(np.float32)  # tp

        for li, p in enumerate(plvls):
            z_base = R_D * t_bar * np.log(1000.0 / p)  # гидростатич. профиль
            z_pert = 300.0 * wave  # крупномасштабная аномалия z (м²/с²)
            z = z_base + z_pert
            arr[t, 4 + li] = z.astype(np.float32)
            arr[t, 17 + li] = (t_bar + 0.05 * (p - 500.0) / 10.0 + 0.5 * wave).astype(np.float32)
            arr[t, 30 + li] = np.clip(40.0 + 20.0 * wave, 5.0, 95.0).astype(np.float32)
            # геострофический ветер из z-аномалии: u=-(1/f)∂z/∂y, v=(1/f)∂z/∂x
            dz_dy = (300.0 * dwave_dlat) / (radius)
            dz_dx = (300.0 * dwave_dlon) / (radius * np.maximum(cphi, 0.05))
            arr[t, 43 + li] = (-dz_dy / f_safe).astype(np.float32)  # u
            arr[t, 56 + li] = (dz_dx / f_safe).astype(np.float32)  # v

    arr.flush()
    meta = {
        "shape": [hours, 69, H, W],
        "dtype": "float32",
        "years": [year],
        "hours_per_year": [hours],
    }
    meta_path = (
        Path(str(out_dat)[:-4] + ".meta.json")
        if str(out_dat).endswith(".dat")
        else out_dat.with_suffix(".meta.json")
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[done] smooth balanced memmap → {out_dat} ({arr.nbytes / 1e6:.1f} MB), meta → {meta_path}"
    )


if __name__ == "__main__":
    out = Path("/tmp/syn_smooth.dat") if len(sys.argv) < 2 else Path(sys.argv[1])
    build_smooth(out)
