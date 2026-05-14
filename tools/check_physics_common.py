"""Общая инфраструктура для CPU-only 72h-rollout проверок физических методов.

Используется тремя check_physics_*.py скриптами:
  * tools/check_physics_weathergft.py     — FD-4 + Euler + const Coriolis
  * tools/check_physics_predformergft.py  — WENO-5 + Euler + beta-plane Coriolis
  * tools/check_physics_weathergft_3.py   — FD-4 + RK4 + spherical + radiation + mixing

Семантика rollout — «чистая физика»: без `scale_diff`, без `.detach()`,
без обучаемых слоёв. Если ECT (атмосфера) разлетится — это и есть результат,
NaN значение логируется в Comet как nan-метрика.

Никаких GPU операций: device='cpu' жёстко, torch.cuda не импортируется.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Channel layout (69-channel filtered ERA5 1.4°)
# =============================================================================
# From example_data/README.md and Data/weatherbench_128.WeatherBench128.variables_list.
# After variables_list-filter:
#   0: t2m, 1: u10, 2: v10, 3: tp                 (surface)
#   4..16: z @ 13 pressure levels                 (geopotential, m²/s²)
#   17..29: t @ 13 levels                         (temperature, K)
#   30..42: r @ 13 levels                         (relative humidity, %)
#   43..55: u @ 13 levels                         (wind, m/s)
#   56..68: v @ 13 levels                         (wind, m/s)
#
# PRESSURE_LEVELS_HPA: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

PRESSURE_LEVELS_HPA: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

SURFACE_VARS = ("t2m", "u10", "v10", "tp")
# `r` (rel humidity) живёт в ERA5 и используется для Magnus-конверсии в `q`,
# но НЕ обновляется в rollout. Из forecast/NaN-метрик исключаем — иначе ловим
# persistence baseline (rmse между init r и truth r at h), а не физический прогноз.
PROGNOSTIC_VARS = ("z", "t", "u", "v", "q")
# Уровни давления для дополнительных per-level physics-метрик.
KEY_PRESSURE_LEVELS_HPA = (500, 850, 1000)
KEY_LEVEL_INDICES = tuple(PRESSURE_LEVELS_HPA.index(p) for p in KEY_PRESSURE_LEVELS_HPA)

CHANNEL_RANGES = {
    "t2m": (0, 1),
    "u10": (1, 2),
    "v10": (2, 3),
    "tp": (3, 4),
    "z": (4, 17),
    "t": (17, 30),
    "r": (30, 43),
    "u": (43, 56),
    "v": (56, 69),
}


def split_channels_69(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Разрезать тензор (B, 69, H, W) на словарь полей.

    Returns:
        dict[var_name] = (B, C_var, H, W). Surface vars имеют C_var=1.
    """
    out = {}
    for var, (a, b) in CHANNEL_RANGES.items():
        out[var] = x[:, a:b]
    return out


def pack_channels_69(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Обратная операция к split_channels_69."""
    return torch.cat(
        [state[v] for v in ("t2m", "u10", "v10", "tp", "z", "t", "r", "u", "v")], dim=1
    )


# =============================================================================
# ERA5 memmap loader (CPU-side; no GPU touch)
# =============================================================================


@dataclass
class MemmapHandle:
    arr: np.memmap
    meta: dict
    row_starts: dict[int, int]
    shape: tuple[int, ...]

    def row_of(self, ts: pd.Timestamp) -> int:
        first_day = pd.to_datetime(f"{ts.year}-01-01 00:00:00")
        hour_in_year = int((ts - first_day).total_seconds() / 3600)
        return self.row_starts[ts.year] + hour_in_year


def open_memmap(memmap_path: str, memmap_meta_path: str | None = None) -> MemmapHandle:
    """Открыть ERA5 memmap read-only и собрать year→row map."""
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
    return MemmapHandle(arr=arr, meta=meta, row_starts=row_starts, shape=shape)


def load_snapshot(
    handle: MemmapHandle, ts: pd.Timestamp, mean: np.ndarray | None, std: np.ndarray | None
) -> torch.Tensor:
    """Загрузить один ERA5-снимок (1, 69, H, W) на CPU.

    Args:
        handle: открытый memmap.
        ts: timestamp.
        mean, std: если переданы — данные предполагаются НОРМАЛИЗОВАННЫМИ в memmap’е,
            и здесь происходит обратное преобразование `raw * std + mean` к физическим
            единицам (как ожидается PDE_kernel’ом). Если None — данные считаются
            уже в физических единицах (v4 memmap).

    Returns:
        torch.Tensor формы (1, 69, H, W), float32, CPU.
    """
    row = handle.row_of(ts)
    raw = np.asarray(handle.arr[row], dtype=np.float32)  # (69, H, W)
    if mean is not None and std is not None:
        raw = raw * std[:, None, None] + mean[:, None, None]
    return torch.from_numpy(raw).unsqueeze(0)  # (1, 69, H, W)


def load_mean_std(path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Загрузить per-channel mean/std (.npy: (2, 69) или (2, 110); .json: example_data format).

    Если data в memmap’е уже raw — передавать пустой путь, тогда вернётся (None, None).
    """
    if not path:
        return None, None
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.shape[1] == 110:
            # Apply variables_list filter (matches Data/weatherbench_128.py:63-69)
            variables_list = [
                0,
                1,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
            ]
            arr = arr[:, variables_list]
        return arr[0], arr[1]
    elif p.suffix == ".json":
        with open(path) as f:
            j = json.load(f)
        return np.asarray(j["mean"], dtype=np.float32), np.asarray(j["std"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported mean_std format: {path}")


# =============================================================================
# Initial conditions: 12 ICs, первое число каждого месяца 2005-го
# =============================================================================


def default_initial_conditions(year: int = 2005, hour: int = 0) -> list[pd.Timestamp]:
    """12 IC: первое число каждого месяца указанного года, в указанный час UTC."""
    return [pd.Timestamp(year=year, month=m, day=1, hour=hour) for m in range(1, 13)]


# =============================================================================
# r → q (specific humidity) via Magnus saturation
# =============================================================================


def magnus_qs(t_kelvin: torch.Tensor, p_hpa: torch.Tensor) -> torch.Tensor:
    """Saturation specific humidity q_s через Magnus formula.

    Args:
        t_kelvin: (B, P, H, W), температура в К.
        p_hpa: (1, P, 1, 1), давление в гПа (НЕ Па).

    Returns:
        q_s: (B, P, H, W), безразмерное.

    Уравнение:
        e_s(T) = 6.112 · exp(17.67 · (T-273.15) / (T-273.15 + 243.5))   [гПа]
        q_s    = 0.622 · e_s / (p - 0.378 · e_s)
    """
    t_c = t_kelvin - 273.15
    e_s = 6.112 * torch.exp(17.67 * t_c / (t_c + 243.5))  # гПа
    return 0.622 * e_s / (p_hpa - 0.378 * e_s)


def relhum_to_specific(
    r_percent: torch.Tensor, t_kelvin: torch.Tensor, p_hpa: torch.Tensor
) -> torch.Tensor:
    """q ≈ (r/100) · q_s(T, p). Простое приближение."""
    return (r_percent / 100.0) * magnus_qs(t_kelvin, p_hpa)


# =============================================================================
# Grid geometry (для f_field, pixel_x, pixel_y, lat_weights). NO buffers — plain tensors on CPU.
# =============================================================================


@dataclass
class GeometryCPU:
    H: int
    W: int
    radius: float = 6371.0 * 1000.0
    pressure_hpa: tuple[int, ...] = PRESSURE_LEVELS_HPA

    latitudes: torch.Tensor = field(init=False)  # (H,) радианы
    pixel_x: torch.Tensor = field(init=False)  # (1, 1, H, 1) метры
    pixel_y: torch.Tensor = field(init=False)  # () метры
    pressure_hpa_t: torch.Tensor = field(init=False)  # (1, 13, 1, 1) гПа
    pressure_pa_t: torch.Tensor = field(init=False)  # (1, 13, 1, 1) Па
    pixel_z: torch.Tensor = field(init=False)  # (1, 13, 1, 1) гПа
    M_z: torch.Tensor = field(init=False)  # (13, 13)
    lat_weights: torch.Tensor = field(init=False)  # (H,)

    def __post_init__(self) -> None:
        H, W = self.H, self.W
        # Линейное распределение широт от -90 до 90, без полюсов.
        lat_deg = torch.linspace(-90.0, 90.0, steps=H + 2)[1:-1]
        self.latitudes = lat_deg / 180.0 * torch.pi

        c_lats = 2 * torch.pi * self.radius * torch.cos(self.latitudes)
        self.pixel_x = (c_lats / W).reshape(1, 1, H, 1)
        self.pixel_y = torch.tensor(torch.pi * self.radius / (H + 1), dtype=torch.float32)

        self.pressure_hpa_t = torch.tensor(self.pressure_hpa, dtype=torch.float32).reshape(
            1, -1, 1, 1
        )
        self.pressure_pa_t = self.pressure_hpa_t * 100.0
        # Δp между уровнями (как в WeatherGFT.py:29)
        pixel_z_values = (50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75)
        self.pixel_z = torch.tensor(pixel_z_values, dtype=torch.float32).reshape(1, -1, 1, 1)

        P = self.pixel_z.shape[1]
        M_z = torch.zeros(P, P)
        for i in range(P):
            for j in range(P):
                if i <= j:
                    M_z[i, j] = self.pixel_z[0, j, 0, 0]
        self.M_z = M_z

        # Latitude weights (cos(lat) normalised, как в `utils.metrics.weighted_rmse_torch`).
        cos_w = torch.cos(self.latitudes)
        s = cos_w.sum()
        self.lat_weights = H * cos_w / s


def integral_z_cpu(field: torch.Tensor, M_z: torch.Tensor) -> torch.Tensor:
    """Кумулятивное интегрирование по давлению, CPU-friendly."""
    B, P, H, W = field.shape
    flat = field.reshape(B, P, H * W)
    out = M_z @ flat
    return out.reshape(B, P, H, W)


# =============================================================================
# Coriolis factories
# =============================================================================


def coriolis_constant(geom: GeometryCPU, value: float = 7.29e-5) -> torch.Tensor:
    """Скалярный f (как в Models/WeatherGFT.py:125)."""
    return torch.tensor(value)


def coriolis_beta_plane(
    geom: GeometryCPU, f0: float = 7.29e-5, beta: float = 1.6e-11
) -> torch.Tensor:
    """f = f0 + β·R·φ (как в Models/PredFormerGFT.py:222-225)."""
    y = geom.radius * geom.latitudes
    return (f0 + beta * y).reshape(1, 1, -1, 1)


def coriolis_spherical(geom: GeometryCPU, omega: float = 7.29e-5) -> torch.Tensor:
    """f = 2Ω sin φ (как в Models/dev/WeatherGFT_3.py:183)."""
    return (2 * omega * torch.sin(geom.latitudes)).reshape(1, 1, -1, 1)


# =============================================================================
# Comet ML logger
# =============================================================================


class Comet72hLogger:
    """Обёртка над comet_ml.Experiment с per-(variable, plvl, lead_hour) логированием.

    Имена метрик (Comet группирует по `/`):
        Forecast (lat-weighted):
            weighted_rmse/<var>/<plvl>hPa     — 5 prog vars × 13 lvls
            weighted_mae/<var>/<plvl>hPa
            weighted_bias/<var>/<plvl>hPa
            acc/<var>/<plvl>hPa               — spatial-Pearson ACC
            weighted_rmse/surface/<var>       — 4 surface (persistence baseline)
            weighted_bias/surface/<var>

        Physics consistency:
            physics/mass_div_abs_mean[/<plvl>hPa]
            physics/ke_mean, physics/ke_max
            physics/geo_u_resid_abs_mean[/<plvl>hPa]
            physics/geo_v_resid_abs_mean[/<plvl>hPa]
            physics/cfl_x_max, physics/cfl_y_max    (по СУБ-STEP dt)
            physics/Q_abs_max                       (root-cause heating term)
            physics/{u_t,v_t,t_t,q_t,z_t}_abs_max   (на каждом step, включая h=0)

        Stability:
            nan_count/<var>                  — счётчик NaN/Inf клеток в state
            frac_ic_blown_up                 — фракция IC, для которых rollout
                                                сломан к моменту h

    step = lead-hour ∈ {0, 1, ..., 72}.
    Среднее по IC — НО: если хоть один IC дал non-finite, итог = NaN
    (раньше тихо пропускалось → метрики между методами выглядели одинаковыми).
    """

    def __init__(
        self,
        project_name: str,
        workspace: str | None,
        api_key: str | None,
        run_name: str,
        offline_dir: str | None = None,
        tags: list[str] | None = None,
    ):
        # NB: импорт comet_ml локальный сюда, чтобы скрипт можно было прогнать
        # с --no-comet даже без установленной библиотеки.
        from comet_ml import Experiment, OfflineExperiment

        if api_key:
            self.experiment = Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                auto_metric_logging=False,
                auto_param_logging=False,
                disabled=False,
            )
        else:
            offline_dir = offline_dir or "logs/comet_offline"
            os.makedirs(offline_dir, exist_ok=True)
            self.experiment = OfflineExperiment(
                project_name=project_name,
                workspace=workspace,
                offline_directory=offline_dir,
                auto_metric_logging=False,
                auto_param_logging=False,
            )

        self.experiment.set_name(run_name)
        if tags:
            for t in tags:
                self.experiment.add_tag(t)

    def log_parameters(self, params: dict) -> None:
        self.experiment.log_parameters(params)

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Залогировать все метрики на одном lead-hour (step=lead_hour)."""
        for name, value in metrics.items():
            if value is None:
                continue
            # NaN/Inf нормально проходят в Comet — он их сериализует. Защищаемся
            # от extreme значений, которые могут сломать UI: clipping для очень
            # больших.
            if isinstance(value, float) and abs(value) > 1e30:
                value = float("nan")
            self.experiment.log_metric(name, value, step=step)

    def end(self) -> None:
        self.experiment.end()


# =============================================================================
# Metrics (per-variable, per-level)
# =============================================================================


def _weighted_stats(
    pred: torch.Tensor, truth: torch.Tensor, lat_w: torch.Tensor
) -> tuple[float, float, float, float]:
    """Lat-weighted RMSE / MAE / signed bias / spatial-anomaly ACC.

    Args:
        pred, truth: одинаковые формы ``(B, H, W)`` или ``(B, 1, H, W)``.
        lat_w: ``(H,)`` — нормированные cos(lat)·H/Σcos.

    Returns:
        (wrmse, wmae, wbias, acc).

    ACC берётся как lat-weighted **spatial-Pearson** между prediction и truth:
    аномалия = поле − его lat-weighted spatial mean (за-IC, on-the-fly), так
    что климатология вычисляется *на ходу* из самого `truth`. Это не каноничный
    ECMWF-ACC (нужна 30-летняя climatology), но валидно для коротких rollout’ов:
    ACC=1 ↔ полностью совпадают пространственные паттерны.
    """
    # Squeeze to (B, H, W) if there's a singleton level/channel axis.
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
        truth = truth.squeeze(1)
    assert pred.dim() == 3, f"Expected (B, H, W), got {pred.shape}"
    B, H, W = pred.shape
    w = lat_w.view(1, H, 1)                              # (1, H, 1)
    w_sum = float(lat_w.sum().item()) * W                # scalar per-sample weight total

    diff = pred - truth
    wmse_sum = (w * diff * diff).sum(dim=(-1, -2))       # (B,)
    wrmse = torch.sqrt((wmse_sum / w_sum).mean()).item()
    wmae = ((w * diff.abs()).sum(dim=(-1, -2)) / w_sum).mean().item()
    wbias = ((w * diff).sum(dim=(-1, -2)) / w_sum).mean().item()

    p_mean = (w * pred).sum(dim=(-1, -2), keepdim=True) / w_sum   # (B, 1, 1)
    t_mean = (w * truth).sum(dim=(-1, -2), keepdim=True) / w_sum
    p_anom = pred - p_mean
    t_anom = truth - t_mean
    num = (w * p_anom * t_anom).sum(dim=(-1, -2))
    den = torch.sqrt(
        (w * p_anom * p_anom).sum(dim=(-1, -2)) * (w * t_anom * t_anom).sum(dim=(-1, -2)) + 1e-30
    )
    acc = (num / den).mean().item()

    return wrmse, wmae, wbias, acc


def compute_forecast_metrics(
    pred_state: dict[str, torch.Tensor],
    truth_state: dict[str, torch.Tensor],
    geom: GeometryCPU,
) -> dict[str, float]:
    """Lat-weighted RMSE / MAE / signed bias / spatial-ACC per (var, level).

    Metric naming (Comet groups by `/`):
        weighted_rmse/surface/<var>          — 4 vars (persistence baseline)
        weighted_bias/surface/<var>          — 4 vars
        weighted_rmse/<var>/<plvl>hPa        — 5 vars × 13 levels = 65
        weighted_mae/<var>/<plvl>hPa         — 65
        weighted_bias/<var>/<plvl>hPa        — 65
        acc/<var>/<plvl>hPa                  — 65

    Удалено vs прежней версии:
        rmse/* (unweighted, дублирует weighted на coarse-grid)
        mae/* (unweighted)
        psnr/* (PSNR для weather полей нерелевантен — dynamic range разнороден)
        weighted_rmse/r/* (persistence — `r` не обновляется в rollout)
    """
    metrics: dict[str, float] = {}
    lat_w = geom.lat_weights

    # Surface — persistence baseline (физика их не трогает).
    for var in SURFACE_VARS:
        wrmse, _, wbias, _ = _weighted_stats(pred_state[var], truth_state[var], lat_w)
        metrics[f"weighted_rmse/surface/{var}"] = wrmse
        metrics[f"weighted_bias/surface/{var}"] = wbias

    # Prognostic per pressure level.
    for var in PROGNOSTIC_VARS:
        p_full = pred_state[var]                          # (B, P, H, W)
        t_full = truth_state[var]
        for lvl_idx, plvl in enumerate(PRESSURE_LEVELS_HPA):
            wrmse, wmae, wbias, acc = _weighted_stats(
                p_full[:, lvl_idx], t_full[:, lvl_idx], lat_w
            )
            metrics[f"weighted_rmse/{var}/{plvl}hPa"] = wrmse
            metrics[f"weighted_mae/{var}/{plvl}hPa"] = wmae
            metrics[f"weighted_bias/{var}/{plvl}hPa"] = wbias
            metrics[f"acc/{var}/{plvl}hPa"] = acc

    return metrics


def compute_physics_metrics(
    state: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor] | None,
    geom: GeometryCPU,
    f_field: torch.Tensor,
    d_x_fn: Callable[[torch.Tensor], torch.Tensor],
    d_y_fn: Callable[[torch.Tensor], torch.Tensor],
    dt_seconds: float,
) -> dict[str, float]:
    """Physics consistency на текущем state.

    Удалено vs прежней версии:
      * physics/pv_abs_mean — псевдо-PV (без ∂θ/∂p), вводящий в заблуждение.
      * physics/mass_div_abs_max — дублирует mean на coarse-grid.

    Добавлено:
      * physics/{mass_div,geo_u_resid,geo_v_resid}/<plvl>hPa
            — per-level срезы для 500/850/1000 hPa (диагностика тропосферы).
      * physics/Q_abs_max
            — log абс. размера диабатического heating Q = -L·z_z·w
              (root cause overflow, см. tools/diagnose_first_substep.py).
      * physics/<rhs>_abs_max — теперь логируется на каждом step (h=0..N).

    CFL теперь считается по СУБ-STEP dt (=block_dt_seconds), а не по часу.
    """
    u, v, z = state["u"], state["v"], state["z"]
    m: dict[str, float] = {}
    n_lvls = u.shape[1]

    div = d_x_fn(u) + d_y_fn(v)                          # mass divergence (1/s)
    div_abs = div.abs()
    m["physics/mass_div_abs_mean"] = div_abs.mean().item()
    for plvl, idx in zip(KEY_PRESSURE_LEVELS_HPA, KEY_LEVEL_INDICES, strict=True):
        if idx < n_lvls:
            m[f"physics/mass_div_abs_mean/{plvl}hPa"] = div_abs[:, idx].mean().item()

    ke = 0.5 * (u * u + v * v)
    m["physics/ke_mean"] = ke.mean().item()
    m["physics/ke_max"] = ke.max().item()

    z_x = d_x_fn(z)
    z_y = d_y_fn(z)
    geo_u = f_field * u + z_y                            # геострофическая невязка по u
    geo_v = f_field * v - z_x                            # и по v
    geo_u_abs = geo_u.abs()
    geo_v_abs = geo_v.abs()
    m["physics/geo_u_resid_abs_mean"] = geo_u_abs.mean().item()
    m["physics/geo_v_resid_abs_mean"] = geo_v_abs.mean().item()
    for plvl, idx in zip(KEY_PRESSURE_LEVELS_HPA, KEY_LEVEL_INDICES, strict=True):
        if idx < n_lvls:
            m[f"physics/geo_u_resid_abs_mean/{plvl}hPa"] = geo_u_abs[:, idx].mean().item()
            m[f"physics/geo_v_resid_abs_mean/{plvl}hPa"] = geo_v_abs[:, idx].mean().item()

    # CFL вычисляется по реальному substep dt, переданному снаружи.
    cfl_x = (u.abs() * dt_seconds / geom.pixel_x).max().item()
    cfl_y = (v.abs() * dt_seconds / geom.pixel_y).max().item()
    m["physics/cfl_x_max"] = cfl_x
    m["physics/cfl_y_max"] = cfl_y

    # Q-term — главный виновник blow-up в legacy формуле. Считаем явно из state,
    # независимо от RHS, чтобы видеть `|Q|` даже когда rollout_step_fn её не отдаёт.
    # Q = -L · z_z · w, где w = integral_z(-(u_x + v_y)). Это repeats работы rhs,
    # но дёшево и единственный надёжный отрезок прямой диагностики формулы.
    L = 2.5e6
    w_z = -(d_x_fn(u) + d_y_fn(v))                       # NB d_x_fn(u) повторно — кеш не делаем (CPU дешёво)
    w = integral_z_cpu(w_z, geom.M_z)
    # z_z по pressure: используем простую FD-2 по pressure axis, чтобы не тянуть d_z_fn.
    # На state[z] shape (B, P, H, W). FD-2: (z[i+1] - z[i-1]) / (p[i+1]-p[i-1]).
    z_pad = torch.cat([z[:, :1], z, z[:, -1:]], dim=1)
    p_pad = geom.pressure_hpa_t                          # (1, P, 1, 1)
    p_diff = torch.cat([p_pad[:, 1:2] - p_pad[:, :1], p_pad[:, 2:] - p_pad[:, :-2], p_pad[:, -1:] - p_pad[:, -2:-1]], dim=1)
    z_z = (z_pad[:, 2:] - z_pad[:, :-2]) / p_diff
    Q = -L * z_z * w
    m["physics/Q_abs_max"] = Q.abs().max().item()

    if rhs is not None:
        for k in ("u_t", "v_t", "t_t", "q_t", "z_t"):
            if k in rhs:
                m[f"physics/{k}_abs_max"] = rhs[k].abs().max().item()

    return m


def count_nans_per_var(state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Количество NaN/Inf на каждый prognostic var. q теперь включён (раньше пропускался)."""
    out: dict[str, float] = {}
    for var in PROGNOSTIC_VARS:                          # ("z", "t", "u", "v", "q") — теперь с q
        s = state[var]
        out[f"nan_count/{var}"] = float(torch.isnan(s).sum().item() + torch.isinf(s).sum().item())
    return out


# =============================================================================
# Generic 72h rollout driver
# =============================================================================


def run_72h_rollout(
    *,
    method_name: str,
    rollout_step_fn: Callable[
        [dict[str, torch.Tensor]], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ],
    d_x_fn: Callable[[torch.Tensor], torch.Tensor],
    d_y_fn: Callable[[torch.Tensor], torch.Tensor],
    f_field: torch.Tensor,
    geom: GeometryCPU,
    initial_conditions: list[pd.Timestamp],
    memmap_path: str,
    memmap_meta_path: str | None,
    mean_std_path: str,
    horizon_hours: int = 72,
    block_dt_seconds: float = 300.0,
    project_name: str = "WeatherPredictions",
    workspace: str | None = None,
    api_key: str | None = None,
    tags: list[str] | None = None,
    offline: bool = False,
) -> None:
    """Прогнать 72h rollout по всем IC, логируя в Comet ML.

    Args:
        method_name: human-readable id метода (например ``weathergft``).
        rollout_step_fn: функция, принимающая текущий state-dict и возвращающая
            (state_next, rhs_dict). RHS опционально (можно вернуть {}).
        d_x_fn, d_y_fn: операторы для physics-consistency метрик.
        f_field: Coriolis.
        geom: геометрия.
        initial_conditions: список pd.Timestamp.
        memmap_path: ERA5.
        memmap_meta_path: meta.json (если None — derived).
        mean_std_path: per-channel mean/std (пустая строка — данные считаются
            уже в физических единицах).
        horizon_hours: горизонт прогноза.
        block_dt_seconds: внутренний шаг substep’а. Substeps per hour =
            3600 / block_dt_seconds (целое).
        project_name, workspace, api_key: Comet creds.
        tags: тэги Comet.
        offline: запись в OfflineExperiment (нужен upload_experiment позже).
    """
    if 3600 % int(block_dt_seconds) != 0:
        raise ValueError(f"3600 must be divisible by block_dt={block_dt_seconds}")
    substeps_per_hour = int(3600 // int(block_dt_seconds))

    # Comet setup
    print(f"[init] Comet experiment for method={method_name}")
    if offline:
        api_key_eff = None
    else:
        api_key_eff = api_key or os.environ.get("COMET_API_KEY")
    workspace_eff = workspace or os.environ.get("COMET_WORKSPACE")
    project_name_eff = project_name or os.environ.get("COMET_PROJECT_NAME") or "WeatherPredictions"

    # Comet experiment name: method first, then context, then timestamp. Это
    # сразу выделяет метод в Comet UI и позволяет фильтровать по prefix’у.
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{method_name}_72h_{timestamp_str}"
    logger = Comet72hLogger(
        project_name=project_name_eff,
        workspace=workspace_eff,
        api_key=api_key_eff,
        run_name=run_name,
        offline_dir=f"logs/comet_offline/{method_name}",
        tags=(tags or []) + [method_name, f"H{geom.H}", f"W{geom.W}", "cpu", "72h"],
    )
    # Дополнительно отдельным метаполем — для group-by в Comet.
    logger.experiment.log_other("method", method_name)
    logger.experiment.log_other("run_timestamp", timestamp_str)

    logger.log_parameters(
        {
            "method": method_name,
            "H": geom.H,
            "W": geom.W,
            "horizon_hours": horizon_hours,
            "block_dt_seconds": block_dt_seconds,
            "substeps_per_hour": substeps_per_hour,
            "n_initial_conditions": len(initial_conditions),
            "initial_conditions": [str(t) for t in initial_conditions],
            "pressure_levels_hpa": list(PRESSURE_LEVELS_HPA),
            "memmap_path": memmap_path,
            "device": "cpu",
        }
    )

    # ERA5
    print(f"[init] Opening memmap {memmap_path}")
    handle = open_memmap(memmap_path, memmap_meta_path)
    print(f"[init] Memmap shape={handle.shape}, years={handle.meta.get('years')}")
    mean, std = load_mean_std(mean_std_path)

    # Aggregator: для каждого (h, metric) храним сумму конечных значений,
    # счётчик конечных IC, и флаг «есть ли non-finite». Финальная семантика —
    # если хоть один IC дал NaN/Inf, логируем NaN (раньше тихо пропускали,
    # из-за чего метрики казались идентичными между методами).
    sum_metrics: dict[int, dict[str, float]] = {h: {} for h in range(horizon_hours + 1)}
    cnt_metrics: dict[int, dict[str, int]] = {h: {} for h in range(horizon_hours + 1)}
    poisoned: dict[int, set[str]] = {h: set() for h in range(horizon_hours + 1)}
    nan_ic_count: dict[int, int] = {h: 0 for h in range(horizon_hours + 1)}

    def _accumulate(h: int, m: dict[str, float]) -> None:
        bucket_s = sum_metrics[h]
        bucket_c = cnt_metrics[h]
        bucket_p = poisoned[h]
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                bucket_p.add(k)
                continue
            bucket_s[k] = bucket_s.get(k, 0.0) + float(v)
            bucket_c[k] = bucket_c.get(k, 0) + 1

    def _rhs_at(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Один substep на копии state, чтобы получить RHS без модификации state."""
        cloned = {k: v.clone() for k, v in state_dict.items()}
        _, rhs_dict = rollout_step_fn(cloned)
        return rhs_dict

    t_start = time.time()
    for ic_idx, ts0 in enumerate(initial_conditions):
        print(f"\n[IC {ic_idx + 1}/{len(initial_conditions)}] {ts0}")
        # Init state в физических единицах.
        x0 = load_snapshot(handle, ts0, mean, std)
        state = _prepare_state(x0, geom)

        # h=0: forecast = identity (RMSE=0), physics ON ERA5 c RHS-baseline.
        # RHS считаем через copy-step, чтобы получить tendency без интегрирования.
        rhs_h0 = _rhs_at(state)
        _accumulate(0, compute_forecast_metrics(state, state, geom))
        _accumulate(
            0,
            compute_physics_metrics(
                state, rhs_h0, geom, f_field, d_x_fn, d_y_fn, dt_seconds=block_dt_seconds
            ),
        )
        _accumulate(0, count_nans_per_var(state))

        ic_blew_up_at: int | None = None
        for hour in range(1, horizon_hours + 1):
            last_rhs: dict[str, torch.Tensor] | None = None
            for _ in range(substeps_per_hour):
                state, last_rhs = rollout_step_fn(state)

            ts_truth = ts0 + pd.Timedelta(hours=hour)
            x_truth = load_snapshot(handle, ts_truth, mean, std)
            truth_state = _prepare_state(x_truth, geom)

            forecast_m = compute_forecast_metrics(state, truth_state, geom)
            physics_m = compute_physics_metrics(
                state, last_rhs, geom, f_field, d_x_fn, d_y_fn, dt_seconds=block_dt_seconds
            )
            nan_m = count_nans_per_var(state)
            _accumulate(hour, forecast_m)
            _accumulate(hour, physics_m)
            _accumulate(hour, nan_m)

            # Отслеживаем «первый час, на котором IC сломался».
            if ic_blew_up_at is None:
                if any(v > 0 for v in nan_m.values()):
                    ic_blew_up_at = hour
                    nan_ic_count[hour] += 1

            if hour % 12 == 0 or hour in (1, 6, 24, 48, 72):
                elapsed = time.time() - t_start
                u_blow = state["u"].abs().max().item()
                wrmse_u500 = forecast_m.get("weighted_rmse/u/500hPa", float("nan"))
                print(
                    f"  h={hour:3d}  |u|max={u_blow:.2e}  "
                    f"wrmse(u@500)={wrmse_u500:.3e}  "
                    f"cfl_x={physics_m['physics/cfl_x_max']:.3e}  elapsed={elapsed:.0f}s"
                )

    # Average across IC. Если хоть один IC дал non-finite на metric+hour,
    # финальное значение = NaN (точная семантика «есть сломанные траектории»).
    print("\n[log] Pushing per-step metrics to Comet (NaN-propagating)…")
    n_ic = len(initial_conditions)
    for h in range(horizon_hours + 1):
        agg: dict[str, float] = {}
        all_keys = set(sum_metrics[h].keys()) | poisoned[h]
        for k in all_keys:
            if k in poisoned[h]:
                agg[k] = float("nan")
            else:
                c = cnt_metrics[h][k]
                agg[k] = sum_metrics[h][k] / c if c > 0 else float("nan")
        # Дополнительная сводка: какая фракция IC уже сломана к этому часу.
        # nan_ic_count считает «впервые сломавшихся на h»; cumulative — сумма
        # по hours ≤ h.
        cum_broken = sum(nan_ic_count[hh] for hh in range(h + 1))
        agg["frac_ic_blown_up"] = cum_broken / float(n_ic)
        logger.log_step(h, agg)

    elapsed_total = time.time() - t_start
    print(f"[done] Elapsed: {elapsed_total:.0f}s")
    logger.experiment.log_metric("wall_seconds_total", elapsed_total)
    logger.end()


def _prepare_state(x: torch.Tensor, geom: GeometryCPU) -> dict[str, torch.Tensor]:
    """Расщепить (1, 69, H, W) на dict, перевести r → q через Magnus."""
    parts = split_channels_69(x)
    q = relhum_to_specific(parts["r"], parts["t"], geom.pressure_hpa_t)
    parts["q"] = q
    return parts


__all__ = [
    "PRESSURE_LEVELS_HPA",
    "SURFACE_VARS",
    "PROGNOSTIC_VARS",
    "CHANNEL_RANGES",
    "split_channels_69",
    "pack_channels_69",
    "MemmapHandle",
    "open_memmap",
    "load_snapshot",
    "load_mean_std",
    "default_initial_conditions",
    "magnus_qs",
    "relhum_to_specific",
    "GeometryCPU",
    "integral_z_cpu",
    "coriolis_constant",
    "coriolis_beta_plane",
    "coriolis_spherical",
    "Comet72hLogger",
    "compute_forecast_metrics",
    "compute_physics_metrics",
    "count_nans_per_var",
    "run_72h_rollout",
]
