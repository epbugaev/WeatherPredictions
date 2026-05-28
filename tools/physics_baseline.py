"""Физический бейзлайн на ERA5 1.4° глобус 2000-2010.

Прогоняет чистый PurePDEKernel (без обучаемых слоёв) `n_substeps` шагов
между ERA5-снимками `(t, t+lead_hours)`, считает per-channel метрики и
физическую consistency, сохраняет в parquet/npy под
`checkpoints/physics_baseline/<tag>/`.

Memmap convention:
    * v4 (default, raw): данные уже в физических единицах. Не передавайте
      `--memmap-is-normalized`; `--mean-std-path` опционален и игнорируется.
    * v3 (legacy normalised): данные хранятся как `(x - mean) / std`.
      Передайте `--memmap-is-normalized` И `--mean-std-path /path/to/mean_std.npy`,
      чтобы код выполнил `x*std + mean` к физическим единицам.

Запуск (cluster, v4 raw):
    python tools/physics_baseline.py \
        --memmap-path /home/ebugaev/era5_memmap/predformer_globe_2000_2018.dat \
        --start-time 2000-01-01 --end-time 2010-12-31 \
        --stride-hours 24 --lead-hours 1 \
        --stencil fd4 --coriolis spherical --time-scheme euler \
        --block-dt 300 --n-substeps 12 \
        --device cuda --batch-size 4 \
        --output-dir checkpoints/physics_baseline/fd4_spherical

Запуск (smoke на синтетике, без ERA5):
    python tools/physics_baseline.py --smoke

Метрики:
    RMSE/MAE/PSNR на 5 prognostic variables (u, v, t, q, z) по уровням
    давления, plus surface T2m/U10/V10 (если включены отдельно).
    Physics consistency:
      * pde_residual: ‖∂X/∂t - RHS(X)‖ для каждой переменной
      * mass_divergence: |∂u/∂x + ∂v/∂y| per pressure level
      * kinetic_energy: 0.5 (u² + v²) — сохраняемость на коротких dt
      * potential_vorticity_proxy: ζ + f
      * geostrophic_residual: |f·v − ∂z/∂x|, |f·u + ∂z/∂y|
      * cfl_number: |u|·dt/dx_min, |v|·dt/dy

Выход:
    <output-dir>/
        config.json        — точные параметры запуска
        metrics.parquet    — сводная таблица по всем timestamps
        residual_stats.npy — гистограммы residuals для отладки
        cfl_max.npy        — max-CFL по timestamps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import (
    Grid,
    GridConfig,
    PurePDEKernel,
    cfl_number,
    geostrophic_residual,
    kinetic_energy_density,
    mass_divergence,
    pde_residual,
    potential_vorticity_proxy,
)

# =============================================================================
# Channel layout (after variables_list filter, 69 channels)
# =============================================================================
# From example_data/README.md:
#   0: t2m (surface), 1: u10, 2: v10, 3: tp
#   4..16: z @ 13 pressure levels [50, 100, 150, 200, 250, 300, 400, 500,
#                                  600, 700, 850, 925, 1000] hPa
#   17..29: t @ 13 levels
#   30..42: r (relative humidity, %) @ 13 levels — fed as q-proxy, см. примечание
#   43..55: u @ 13 levels
#   56..68: v @ 13 levels

SURFACE_CHANNELS = (0, 1, 2, 3)  # t2m, u10, v10, tp
Z_RANGE = (4, 17)
T_RANGE = (17, 30)
R_RANGE = (30, 43)
U_RANGE = (43, 56)
V_RANGE = (56, 69)


@dataclass
class BaselineConfig:
    memmap_path: str
    memmap_meta_path: str | None
    mean_std_path: str
    start_time: str
    end_time: str
    stride_hours: int
    lead_hours: int
    stencil: Literal["fd4", "weno5"]
    coriolis: Literal["constant", "beta_plane", "spherical"]
    time_scheme: Literal["euler", "rk4"]
    block_dt: float
    n_substeps: int
    boundary_x: Literal["periodic", "reflect", "replicate"]
    boundary_y: Literal["periodic", "reflect", "replicate"]
    boundary_z: Literal["reflect", "replicate"]
    use_universal_R: bool
    device: str
    batch_size: int
    output_dir: str
    H: int
    W: int
    lat_range_deg: tuple[float, float]
    memmap_is_normalized: bool


# =============================================================================
# Data loading
# =============================================================================


def load_memmap_view(
    memmap_path: str, memmap_meta_path: str | None
) -> tuple[np.memmap, dict, dict[int, int]]:
    """Открыть memmap read-only и собрать year -> first-row offset map.

    Returns:
        (memmap, meta, year_to_first_row).
    """
    if memmap_meta_path is None:
        memmap_meta_path = (
            memmap_path[:-4] + ".meta.json"
            if memmap_path.endswith(".dat")
            else memmap_path + ".meta.json"
        )
    with open(memmap_meta_path) as f:
        meta = json.load(f)

    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    arr = np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)

    row_starts: dict[int, int] = {}
    offset = 0
    for y, n in zip(meta["years"], meta["hours_per_year"], strict=True):
        row_starts[int(y)] = offset
        offset += int(n)

    return arr, meta, row_starts


def timestamp_to_row(ts: pd.Timestamp, row_starts: dict[int, int]) -> int:
    """Hourly memmap row для конкретного timestamp."""
    first_day = pd.to_datetime(f"{ts.year}-01-01 00:00:00")
    hour_in_year = int((ts - first_day).total_seconds() / 3600)
    return row_starts[ts.year] + hour_in_year


def load_mean_std(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Загрузить per-channel mean/std (69 значений каждый)."""
    if path.endswith(".npy"):
        m = np.load(path)
        return m[0].astype(np.float32), m[1].astype(np.float32)
    elif path.endswith(".json"):
        with open(path) as f:
            j = json.load(f)
        return np.asarray(j["mean"], dtype=np.float32), np.asarray(j["std"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported mean_std format: {path}")


def split_channels_69(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Расщепить тензор (B, 69, H, W) на словарь полей.

    Returns:
        dict с ключами:
            'surface': (B, 4, H, W) — t2m/u10/v10/tp.
            'z', 't', 'r', 'u', 'v': каждый (B, 13, H, W).
    """
    return {
        "surface": x[:, : SURFACE_CHANNELS[-1] + 1],
        "z": x[:, Z_RANGE[0] : Z_RANGE[1]],
        "t": x[:, T_RANGE[0] : T_RANGE[1]],
        "r": x[:, R_RANGE[0] : R_RANGE[1]],
        "u": x[:, U_RANGE[0] : U_RANGE[1]],
        "v": x[:, V_RANGE[0] : V_RANGE[1]],
    }


def relative_to_specific_humidity(r_percent: torch.Tensor, q_s: torch.Tensor) -> torch.Tensor:
    """Конвертация relative humidity (%) → specific humidity (кг/кг).

    Простое приближение: q ≈ r/100 · q_s, где q_s — saturation specific humidity
    из Magnus-формулы. Не учитывает плотностные поправки.
    """
    return (r_percent / 100.0) * q_s


# =============================================================================
# Metrics
# =============================================================================


def per_channel_rmse_mae(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (axis 1) RMSE и MAE, усреднённые по (B, H, W).

    Args:
        pred, target: (B, C, H, W).

    Returns:
        rmse, mae: (C,).
    """
    diff = pred - target
    rmse = torch.sqrt((diff**2).mean(dim=(0, 2, 3)))
    mae = diff.abs().mean(dim=(0, 2, 3))
    return rmse, mae


def per_channel_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR на канал. data_range — реальный диапазон target’а на этом канале.

    Args:
        pred, target: (B, C, H, W).

    Returns:
        psnr: (C,) в дБ.
    """
    mse = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    target_range = target.amax(dim=(0, 2, 3)) - target.amin(dim=(0, 2, 3))
    target_range = torch.where(target_range > 1e-12, target_range, torch.ones_like(target_range))
    return 20.0 * torch.log10(target_range / torch.sqrt(mse + 1e-12))


def latitude_weighted_rmse(
    pred: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor
) -> torch.Tensor:
    """Latitude-weighted RMSE (cos(lat)-нормированные веса, как `utils.metrics.weighted_rmse_torch`).

    Args:
        pred, target: (B, C, H, W).
        lat_weights: (H,) — нормированные cos(lat).

    Returns:
        rmse: (C,).
    """
    w = lat_weights.view(1, 1, -1, 1)
    diff2 = w * (pred - target) ** 2
    rmse = torch.sqrt(diff2.mean(dim=(0, 2, 3)))
    return rmse


# =============================================================================
# Main loop
# =============================================================================


def run_baseline(cfg: BaselineConfig, smoke: bool = False) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build grid + kernel ---
    grid = Grid(GridConfig(H=cfg.H, W=cfg.W, lat_range_deg=tuple(cfg.lat_range_deg))).to(cfg.device)
    kernel = PurePDEKernel(
        grid,
        stencil=cfg.stencil,
        coriolis=cfg.coriolis,
        block_dt=cfg.block_dt,
        time_scheme=cfg.time_scheme,
        boundary_x=cfg.boundary_x,
        boundary_y=cfg.boundary_y,
        boundary_z=cfg.boundary_z,
        use_universal_R=cfg.use_universal_R,
    ).to(cfg.device)
    print(f"[init] Grid H={cfg.H}, W={cfg.W}; kernel={kernel}")

    # --- latitude weights for weighted RMSE ---
    # B-1 fix: используем grid.latitudes напрямую (linspace(-90, 90, H+2)[1:-1]),
    # чтобы совпадало с check_physics_common.py и не было расхождения на крае.
    cos_w = torch.cos(grid.latitudes)
    lat_weights = cfg.H * cos_w / cos_w.sum()

    if smoke:
        print("[smoke] Synthetic random ERA5-like state, single batch, no memmap.")
        x_now = torch.randn(cfg.batch_size, 69, cfg.H, cfg.W, device=cfg.device)
        x_next = x_now + 0.01 * torch.randn_like(x_now)
        timestamps = pd.date_range("2000-01-01", periods=cfg.batch_size, freq="h")
        rows_now = list(range(cfg.batch_size))
    else:
        print(f"[init] Opening memmap {cfg.memmap_path}")
        arr, meta, row_starts = load_memmap_view(cfg.memmap_path, cfg.memmap_meta_path)
        T_total, C, H_data, W_data = arr.shape
        print(f"[init] Memmap shape: T={T_total}, C={C}, H={H_data}, W={W_data}")
        assert C == 69, f"Expected 69 channels, got {C}"
        assert (H_data, W_data) == (cfg.H, cfg.W), (
            f"Grid mismatch: memmap {H_data}x{W_data} vs cfg {cfg.H}x{cfg.W}"
        )

        # Build sample timestamps grid (start..end, stride=stride_hours)
        start = pd.to_datetime(cfg.start_time)
        end = pd.to_datetime(cfg.end_time)
        timestamps = pd.date_range(start, end, freq=f"{cfg.stride_hours}h")
        rows_now = [timestamp_to_row(t, row_starts) for t in timestamps]
        rows_next = [
            timestamp_to_row(t + pd.Timedelta(hours=cfg.lead_hours), row_starts) for t in timestamps
        ]
        # filter out-of-range
        valid_mask = [
            r < T_total and rn < T_total for r, rn in zip(rows_now, rows_next, strict=True)
        ]
        timestamps = timestamps[valid_mask]
        rows_now = [r for r, m in zip(rows_now, valid_mask, strict=True) if m]
        rows_next = [r for r, m in zip(rows_next, valid_mask, strict=True) if m]
        print(f"[init] {len(timestamps)} valid samples after stride/lead filtering.")

        # Нормализация: денормализуем `raw * std + mean` к физическим единицам
        # ТОЛЬКО если memmap хранит уже нормализованные данные (v3 convention) —
        # это явно отмечено флагом `memmap_is_normalized`. Для v4 raw memmap
        # денормализация ничего не делает только если mean=0, std=1, что не так
        # для ERA5 — пропуск этого шага раньше «бесшумно» портил данные.
        if cfg.memmap_is_normalized:
            if not cfg.mean_std_path:
                raise ValueError(
                    "memmap_is_normalized=True requires --mean-std-path (нужен для "
                    "обратного преобразования к физ. единицам)."
                )
            the_mean, the_std = load_mean_std(cfg.mean_std_path)
            the_mean_t = torch.from_numpy(the_mean).to(cfg.device).view(1, -1, 1, 1)
            the_std_t = torch.from_numpy(the_std).to(cfg.device).view(1, -1, 1, 1)
        else:
            the_mean_t = None
            the_std_t = None

    # --- Iterate in batches ---
    n_samples = cfg.batch_size if smoke else len(timestamps)
    n_batches = (n_samples + cfg.batch_size - 1) // cfg.batch_size
    print(f"[run] {n_samples} samples in {n_batches} batches of {cfg.batch_size}")

    records: list[dict] = []

    t_start = time.time()
    for bi in range(n_batches):
        sl = slice(bi * cfg.batch_size, (bi + 1) * cfg.batch_size)
        if smoke:
            x_now_raw = x_now
            x_next_raw = x_next
            ts_batch = timestamps[sl]
        else:
            rows_now_b = rows_now[sl]
            rows_next_b = rows_next[sl]
            ts_batch = timestamps[sl]
            x_now_raw = (
                torch.from_numpy(np.stack([arr[r] for r in rows_now_b], axis=0))
                .float()
                .to(cfg.device)
            )
            x_next_raw = (
                torch.from_numpy(np.stack([arr[r] for r in rows_next_b], axis=0))
                .float()
                .to(cfg.device)
            )
            # Денормализация только если флаг `--memmap-is-normalized` (v3
            # convention). Дефолт — v4 raw, ничего не делать.
            if the_mean_t is not None:
                x_now_raw = x_now_raw * the_std_t + the_mean_t
                x_next_raw = x_next_raw * the_std_t + the_mean_t

        now = split_channels_69(x_now_raw)
        nxt = split_channels_69(x_next_raw)

        # r -> q proxy (через Magnus от текущего T)
        # NOTE: kernel._get_qs ожидает (p, T). Восстановим p как сетку давления (Па).
        with torch.no_grad():
            pressure_pa = grid.pressure  # (1, 13, 1, 1), в Па
            q_s_now = kernel._get_qs(pressure_pa.expand_as(now["t"]), now["t"])
            q_s_nxt = kernel._get_qs(pressure_pa.expand_as(nxt["t"]), nxt["t"])
            q_now = relative_to_specific_humidity(now["r"], q_s_now)
            q_nxt = relative_to_specific_humidity(nxt["r"], q_s_nxt)

        state_now = {"u": now["u"], "v": now["v"], "t": now["t"], "q": q_now, "z": now["z"]}
        state_truth = {"u": nxt["u"], "v": nxt["v"], "t": nxt["t"], "q": q_nxt, "z": nxt["z"]}

        # --- Multi-substep Euler/RK forward ---
        with torch.no_grad():
            cur = state_now
            for _ in range(cfg.n_substeps):
                step = kernel.step(cur["u"], cur["v"], cur["t"], cur["q"], cur["z"])
                cur = {
                    "u": step["u"],
                    "v": step["v"],
                    "t": step["t"],
                    "q": step["q"],
                    "z": step["z"],
                }
            state_pred = cur

            # Forecast metrics — слим до weighted_rmse + spatial-anomaly ACC
            # (по запросу пользователя). Реализованы здесь локально, чтобы не
            # тянуть check_physics_common (это отдельный entry-point).
            for var in ("u", "v", "t", "q", "z"):
                pred_full = state_pred[var]  # (B, P, H, W)
                truth_full = state_truth[var]
                P = pred_full.shape[1]
                wgrid = lat_weights.view(1, -1, 1)  # (1, H, 1)
                w_sum = float(lat_weights.sum().item()) * cfg.W
                for lvl in range(P):
                    p_2d = pred_full[:, lvl]  # (B, H, W)
                    t_2d = truth_full[:, lvl]
                    diff = p_2d - t_2d
                    wmse = (wgrid * diff * diff).sum(dim=(-1, -2)) / w_sum
                    wrmse = float(torch.sqrt(wmse.mean()).cpu())
                    p_mean = (wgrid * p_2d).sum(dim=(-1, -2), keepdim=True) / w_sum
                    t_mean = (wgrid * t_2d).sum(dim=(-1, -2), keepdim=True) / w_sum
                    p_anom = p_2d - p_mean
                    t_anom = t_2d - t_mean
                    num = (wgrid * p_anom * t_anom).sum(dim=(-1, -2))
                    den = torch.sqrt(
                        (wgrid * p_anom * p_anom).sum(dim=(-1, -2))
                        * (wgrid * t_anom * t_anom).sum(dim=(-1, -2))
                        + 1e-30
                    )
                    acc = float((num / den).mean().cpu())
                    for ts in ts_batch:
                        records.append(
                            {
                                "timestamp": ts.isoformat(),
                                "variable": var,
                                "pressure_level_hpa": int(grid.config.pressure_levels[lvl]),
                                "weighted_rmse": wrmse,
                                "acc": acc,
                            }
                        )

        if bi % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t_start
            rate = (bi + 1) * cfg.batch_size / max(elapsed, 1e-6)
            print(f"[batch {bi + 1}/{n_batches}] {rate:.1f} samples/s")

    # --- Save outputs ---
    df = pd.DataFrame(records)
    df.to_parquet(out_dir / "metrics.parquet", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    elapsed_total = time.time() - t_start
    print(f"[done] {len(df)} records → {out_dir / 'metrics.parquet'}")
    print(f"[done] Elapsed: {elapsed_total:.1f}s")
    # Summary: median weighted_rmse и ACC по 500hPa уровню как индикатор pipeline.
    z500 = df[(df["variable"] == "z") & (df["pressure_level_hpa"] == 500)]
    if not z500.empty:
        print(
            f"[done] z@500hPa median weighted_rmse={z500['weighted_rmse'].median():.3e}  "
            f"ACC={z500['acc'].median():.3f}"
        )


def parse_args(argv: list[str] | None = None) -> tuple[BaselineConfig, bool]:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--memmap-path", default="/home/ebugaev/era5_memmap/predformer_globe_2000_2018.dat"
    )
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument(
        "--mean-std-path",
        default="",
        help="Пустая строка → memmap уже в физических единицах (v4 convention). "
        "Передавайте путь только если memmap содержит нормализованные данные (v3).",
    )
    p.add_argument("--start-time", default="2000-01-01 00:00:00")
    p.add_argument("--end-time", default="2010-12-31 23:00:00")
    p.add_argument("--stride-hours", type=int, default=24)
    p.add_argument("--lead-hours", type=int, default=1)
    p.add_argument("--stencil", choices=["fd4", "weno5"], default="fd4")
    p.add_argument(
        "--coriolis", choices=["constant", "beta_plane", "spherical"], default="spherical"
    )
    p.add_argument("--time-scheme", choices=["euler", "rk4"], default="euler")
    p.add_argument("--block-dt", type=float, default=300.0)
    p.add_argument("--n-substeps", type=int, default=12)
    p.add_argument(
        "--boundary-x",
        choices=["periodic", "reflect", "replicate"],
        default="periodic",
        help="Граничное условие по lon. Дефолт 'periodic'.",
    )
    p.add_argument(
        "--boundary-y",
        choices=["periodic", "reflect", "replicate"],
        default="replicate",
        help="Граничное условие по lat. Дефолт 'replicate' (lat не циклична).",
    )
    p.add_argument(
        "--boundary-z",
        choices=["reflect", "replicate"],
        default="replicate",
        help="Граничное условие по pressure. 'periodic' запрещён.",
    )
    p.add_argument(
        "--memmap-is-normalized",
        action="store_true",
        help="Если True, memmap хранит per-channel-нормализованные данные (v3 convention) "
        "и будет денормализован через --mean-std-path к физическим единицам. "
        "Дефолт False — memmap считается raw (v4 convention).",
    )
    p.add_argument(
        "--use-universal-R",
        action="store_true",
        help="Использовать универсальную R=8.314 Дж/(моль·К) в гидростатике вместо "
        "дефолтного R_d=287 Дж/(кг·К). Opt-in для регрессии старой физики.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", default="checkpoints/physics_baseline/default")
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=256)
    p.add_argument(
        "--lat-range-deg",
        nargs=2,
        type=float,
        default=[-90.0, 90.0],
        metavar=("LOW", "HIGH"),
        help="Диапазон широт в градусах (low high). Дефолт (-90 90) — global. "
        "Для USA-кропа: --lat-range-deg 24 56.",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Run on synthetic random tensor, skip memmap."
    )
    args = p.parse_args(argv)

    smoke = bool(args.smoke)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU.")
        args.device = "cpu"

    cfg = BaselineConfig(
        memmap_path=args.memmap_path,
        memmap_meta_path=args.memmap_meta_path,
        mean_std_path=args.mean_std_path,
        start_time=args.start_time,
        end_time=args.end_time,
        stride_hours=args.stride_hours,
        lead_hours=args.lead_hours,
        stencil=args.stencil,
        coriolis=args.coriolis,
        time_scheme=args.time_scheme,
        block_dt=args.block_dt,
        n_substeps=args.n_substeps,
        boundary_x=args.boundary_x,
        boundary_y=args.boundary_y,
        boundary_z=args.boundary_z,
        use_universal_R=args.use_universal_R,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        H=args.H,
        W=args.W,
        lat_range_deg=tuple(args.lat_range_deg),
        memmap_is_normalized=args.memmap_is_normalized,
    )
    return cfg, smoke


if __name__ == "__main__":
    cfg, smoke = parse_args()
    run_baseline(cfg, smoke=smoke)
