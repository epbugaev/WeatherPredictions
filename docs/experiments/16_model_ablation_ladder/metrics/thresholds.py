"""Пороги событий CSI/FSS: квантили наблюдённого поля на валидации.

У ERA5 нет отражаемости в dBZ, а 69 каналов живут в несопоставимых единицах (z в
м²/с², r в %, ветер в м/с), поэтому абсолютные пороги вроде «20/30/40» не переносятся
между переменными. Берём квантили **истины** (p90/p95/p99 на канал): порог получает
физический смысл «верхние 10/5/1 % значений этого поля», CSI становится сравнимым
между переменными, а единицы перестают мешать.

Пороги считаются один раз по валидационному году и **общие для всех армов** — иначе
армы сравнивались бы на разных множествах событий.

Запуск (кластер, CPU):
    REPO_ROOT=~/wt_fix_v2 python thresholds.py \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --out ~/era5_memmap/thresholds_usa_2004.npz

Выход npz: ``thresholds`` (69, 3) float32, ``quantiles`` (3,), ``channels`` (69,).
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.90, 0.95, 0.99)


def val_frame_mask(meta: dict, val_year: int) -> np.ndarray:
    """Маска кадров валидационного года в packed-memmap."""
    start = pd.Timestamp(f"{meta['start_year']}-01-01 00:00:00")
    times = pd.date_range(start=start, periods=int(sum(meta["hours_per_year"])), freq="1h")
    return times.year.to_numpy() == val_year


def channel_quantiles(
    memmap: np.memmap, mask: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Квантили каждого канала по всем кадрам валидации и всем ячейкам.

    Args:
        memmap: ``(N, C, H, W)`` float32.
        mask: маска кадров валидации, ``(N,)``.
        quantiles: уровни квантилей.

    Returns:
        ``(C, K)`` float32 — пороги в физических единицах.
    """
    rows = np.flatnonzero(mask)
    n_channels = memmap.shape[1]
    result = np.empty((n_channels, len(quantiles)), dtype=np.float32)
    # По каналам, а не одним куском: весь валидационный год — ~5 ГБ, а np.quantile
    # внутри сортирует копию, и пик памяти удваивается.
    for channel in range(n_channels):
        values = np.asarray(memmap[rows, channel], dtype=np.float32).reshape(-1)
        result[channel] = np.quantile(values, quantiles)
    return result


def build_thresholds(args: Namespace) -> None:
    """Посчитать пороги и записать npz. Side effects: читает memmap, пишет ``args.out``."""
    meta_path = Path(str(Path(args.memmap).with_suffix("")) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    memmap = np.memmap(args.memmap, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
    mask = val_frame_mask(meta, args.val_year)
    print(f"[thr] кадров валидации {args.val_year}: {int(mask.sum())}")  # noqa: T201

    thresholds = channel_quantiles(memmap, mask, QUANTILES)
    channels = np.array(meta.get("channel_names", [f"c{i}" for i in range(thresholds.shape[0])]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        thresholds=thresholds,
        quantiles=np.array(QUANTILES),
        channels=channels,
        val_year=args.val_year,
    )
    print(f"[thr] записано {out_path} {thresholds.shape}")  # noqa: T201


def parse_args() -> Namespace:
    """CLI: packed-memmap, выходной npz, год валидации."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--memmap", required=True, help="packed memmap .dat")
    parser.add_argument("--out", required=True, help="выходной .npz с порогами")
    parser.add_argument(
        "--val-year", type=int, default=2004, help="год, по которому берём квантили"
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_thresholds(parse_args())
