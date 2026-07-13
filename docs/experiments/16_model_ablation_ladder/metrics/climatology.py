"""Климатология USA-кропа по дню года из train-лет — база отсчёта для ACC.

ACC меряет корреляцию **аномалий** (поле минус климатология), поэтому без
климатологии он не определён, а в проекте её не было. Считаем её честно: среднее по
train-годам (2000–2003) для каждого дня года, каждой ячейки и канала, со сглаживанием
скользящим окном ±``SMOOTH_DAYS`` дней — иначе при четырёх годах день-в-день шумит.

Валидационный год (2004) в климатологию **не входит** — иначе ACC подсматривал бы
ответ.

Запуск (кластер, CPU):
    REPO_ROOT=~/wt_fix_v2 python climatology.py \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --out ~/era5_memmap/climatology_usa_2000_2003.npz

Выход npz: ``climatology`` (366, 69, 32, 64) float32, ``train_years``, ``smooth_days``.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

SMOOTH_DAYS = 7  # половина окна сглаживания по дню года
DAYS_IN_YEAR = 366


def hourly_timestamps(meta: dict) -> pd.DatetimeIndex:
    """Почасовой календарь, покрывающий все кадры memmap.

    Args:
        meta: содержимое ``<stem>.meta.json`` (ключи ``start_year``, ``hours_per_year``).

    Returns:
        ``pd.DatetimeIndex`` длины ``sum(hours_per_year)``.
    """
    start = pd.Timestamp(f"{meta['start_year']}-01-01 00:00:00")
    return pd.date_range(start=start, periods=int(sum(meta["hours_per_year"])), freq="1h")


def train_frame_mask(times: pd.DatetimeIndex, train_years: tuple[int, ...]) -> np.ndarray:
    """Булева маска кадров, попадающих в train-годы (валидационный год исключён)."""
    return np.isin(times.year.to_numpy(), np.asarray(train_years))


def day_of_year_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Индекс дня года 0..365.

    29 февраля получает собственный слот; в невисокосные годы он остаётся пустым и
    заполняется сглаживанием по соседним дням.
    """
    return times.dayofyear.to_numpy() - 1


def accumulate_daily_means(
    memmap: np.memmap, doy: np.ndarray, mask: np.ndarray, n_channels: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Среднее поле на каждый день года по train-кадрам.

    Args:
        memmap: ``(N, C, H, W)`` float32 — весь packed-memmap.
        doy: индекс дня года на кадр, ``(N,)``.
        mask: маска train-кадров, ``(N,)``.
        n_channels: число каналов C.
        height: H.
        width: W.

    Returns:
        ``(daily (366, C, H, W) float32, counts (366,) int64)`` — сырое (несглаженное)
        среднее по дню года и число кадров в каждом дне. Дни без данных остаются
        нулями с нулевым счётчиком и заполняются на этапе сглаживания.
    """
    totals = np.zeros((DAYS_IN_YEAR, n_channels, height, width), dtype=np.float64)
    counts = np.zeros(DAYS_IN_YEAR, dtype=np.int64)
    for frame_idx in np.flatnonzero(mask):
        day = doy[frame_idx]
        totals[day] += memmap[frame_idx]
        counts[day] += 1
    nonempty = counts > 0
    totals[nonempty] /= counts[nonempty][:, None, None, None]
    return totals.astype(np.float32), counts


def smooth_over_days(daily: np.ndarray, counts: np.ndarray, half_window: int) -> np.ndarray:
    """Сгладить климатологию скользящим окном ±``half_window`` дней (циклично по году).

    Взвешивание по числу кадров в дне: дни без данных (29 февраля в невисокосные годы)
    вклада не дают и заполняются соседями.

    Args:
        daily: ``(366, C, H, W)`` — сырые дневные средние.
        counts: ``(366,)`` — сколько кадров попало в каждый день.
        half_window: половина окна в днях.

    Returns:
        ``(366, C, H, W)`` float32.
    """
    offsets = np.arange(-half_window, half_window + 1)
    weighted_sum = np.zeros_like(daily, dtype=np.float64)
    weight_total = np.zeros(DAYS_IN_YEAR, dtype=np.float64)
    for offset in offsets:
        rolled = np.roll(daily, offset, axis=0)
        rolled_counts = np.roll(counts, offset)
        weighted_sum += rolled * rolled_counts[:, None, None, None]
        weight_total += rolled_counts
    return (weighted_sum / weight_total[:, None, None, None]).astype(np.float32)


def build_climatology(args: Namespace) -> None:
    """Посчитать климатологию по дню года и записать npz.

    Side effects: читает memmap, пишет ``args.out``, печатает прогресс.
    """
    meta_path = Path(str(Path(args.memmap).with_suffix("")) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    n_frames, n_channels, height, width = meta["shape"]

    times = hourly_timestamps(meta)
    assert len(times) == n_frames, f"календарь {len(times)} != кадров {n_frames}"

    train_years = tuple(year for year in meta["years"] if year != args.val_year)
    mask = train_frame_mask(times, train_years)
    doy = day_of_year_index(times)
    print(f"[clim] train-годы {train_years}, кадров {int(mask.sum())} из {n_frames}")  # noqa: T201

    memmap = np.memmap(args.memmap, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
    daily, counts = accumulate_daily_means(memmap, doy, mask, n_channels, height, width)
    print(f"[clim] дневных средних: {int((counts > 0).sum())}/366 дней заполнено")  # noqa: T201

    climatology = smooth_over_days(daily, counts, SMOOTH_DAYS)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        climatology=climatology,
        train_years=np.array(train_years),
        smooth_days=SMOOTH_DAYS,
        val_year=args.val_year,
    )
    print(f"[clim] записано {out_path} {climatology.shape}")  # noqa: T201


def parse_args() -> Namespace:
    """CLI: packed-memmap, выходной npz, валидационный год (исключается из климатологии)."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--memmap", required=True, help="packed memmap .dat")
    parser.add_argument("--out", required=True, help="выходной .npz с климатологией")
    parser.add_argument("--val-year", type=int, default=2004, help="год валидации (исключается)")
    return parser.parse_args()


if __name__ == "__main__":
    build_climatology(parse_args())
