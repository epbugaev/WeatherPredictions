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
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from utils.physics import PurePDEKernel


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


def magnus_qs(t_kelvin: torch.Tensor, p_pa: torch.Tensor) -> torch.Tensor:
    """Saturation specific humidity q_s через Magnus formula (SI: давление в Па).

    Args:
        t_kelvin: (B, P, H, W), температура в К.
        p_pa: (1, P, 1, 1), давление в Па (НЕ гПа). Согласовано с
            ``utils.physics.PurePDEKernel._get_qs`` и ``utils.physics.Grid.pressure``.

    Returns:
        q_s: (B, P, H, W), безразмерное (кг/кг).

    Уравнение:
        e_s(T) = 611.2 · exp(17.67 · (T-273.15) / (T-273.15 + 243.5))   [Па]
        q_s    = 0.622 · e_s / (p_pa - 0.378 · e_s)
    """
    t_c = t_kelvin - 273.15
    e_s = 611.2 * torch.exp(17.67 * t_c / (t_c + 243.5))  # Па
    return 0.622 * e_s / (p_pa - 0.378 * e_s)


def relhum_to_specific(
    r_percent: torch.Tensor, t_kelvin: torch.Tensor, p_pa: torch.Tensor
) -> torch.Tensor:
    """q ≈ (r/100) · q_s(T, p_pa). Простое приближение."""
    return (r_percent / 100.0) * magnus_qs(t_kelvin, p_pa)


# =============================================================================
# Adiabatic temperature tendency (correct dT/dt|_adia = α·ω/c_p)
# =============================================================================


def adiabatic_temperature_tendency(
    t_kelvin: torch.Tensor,
    w_legacy_hpa_s: torch.Tensor,
    p_pa: torch.Tensor,
    r_d: float = 287.0,
    c_p: float = 1005.0,
) -> torch.Tensor:
    """Адиабатическая температура: dT/dt|_adia = α·ω / c_p = R_d·T·ω / (c_p·p).

    Заменяет сломанную ``Q = -L·z_z·w`` формулу из старого WeatherGFT.py.
    Принимает «legacy» вертикальную скорость w в hPa/s (как возвращает
    `integral_z(-(u_x+v_y))` на сетке с pixel_z в гПа) и конвертирует её в
    ω (Pa/s) фактором ×100.

    Args:
        t_kelvin: (B, P, H, W) температура.
        w_legacy_hpa_s: (B, P, H, W) вертикальная скорость в **hPa/s**.
        p_pa: (1, P, 1, 1) или broadcastable, давление в Па.
        r_d: газовая постоянная сухого воздуха (J/(kg·K)).
        c_p: теплоёмкость при постоянном давлении (J/(kg·K)).

    Returns:
        dT/dt|_adia в К/с, та же форма что и t_kelvin.
    """
    omega_pa = 100.0 * w_legacy_hpa_s
    return r_d * t_kelvin * omega_pa / (c_p * p_pa)


# =============================================================================
# Grid geometry (для f_field, pixel_x, pixel_y, lat_weights). NO buffers — plain tensors on CPU.
# =============================================================================


@dataclass
class GeometryCPU:
    H: int
    W: int
    radius: float = 6371.0 * 1000.0
    pressure_hpa: tuple[int, ...] = PRESSURE_LEVELS_HPA
    lat_range_deg: tuple[float, float] = (-90.0, 90.0)

    latitudes: torch.Tensor = field(init=False)  # (H,) радианы
    pixel_x: torch.Tensor = field(init=False)  # (1, 1, H, 1) метры
    pixel_y: torch.Tensor = field(init=False)  # () метры
    pressure_pa_t: torch.Tensor = field(init=False)  # (1, 13, 1, 1) Па (SI)
    pixel_z: torch.Tensor = field(init=False)  # (1, 13, 1, 1) гПа (Δp)
    M_z: torch.Tensor = field(init=False)  # (13, 13)
    lat_weights: torch.Tensor = field(init=False)  # (H,)

    def __post_init__(self) -> None:
        H, W = self.H, self.W
        lat_low, lat_high = self.lat_range_deg
        if not (-90.0 <= lat_low < lat_high <= 90.0):
            raise ValueError(
                f"Invalid lat_range_deg {self.lat_range_deg!r}: expected "
                "(low, high) with -90 ≤ low < high ≤ 90."
            )
        # Линейное распределение широт в заданном диапазоне, без крайних точек.
        lat_deg = torch.linspace(lat_low, lat_high, steps=H + 2)[1:-1]
        self.latitudes = lat_deg / 180.0 * torch.pi

        c_lats = 2 * torch.pi * self.radius * torch.cos(self.latitudes)
        self.pixel_x = (c_lats / W).reshape(1, 1, H, 1)
        lat_span_rad = (lat_high - lat_low) / 180.0 * torch.pi
        self.pixel_y = torch.tensor(lat_span_rad * self.radius / (H + 1), dtype=torch.float32)

        pressure_hpa = torch.tensor(self.pressure_hpa, dtype=torch.float32).reshape(1, -1, 1, 1)
        self.pressure_pa_t = pressure_hpa * 100.0
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


# =============================================================================
# Coriolis factories
# =============================================================================


_OMEGA_EARTH = 7.2921e-5  # рад/с — угловая скорость вращения Земли
_F_MID_LATITUDE = 2 * _OMEGA_EARTH * float(torch.sin(torch.tensor(torch.pi / 4)))  # ≈ 1.03e-4


def coriolis_constant(geom: GeometryCPU, value: float = _F_MID_LATITUDE) -> torch.Tensor:
    """Скалярный Coriolis. **Дефолт функции** = 2Ω·sin(45°) ≈ 1.03e-4 (каноничный).

    ВАЖНО: возвращаемое значение определяется аргументом `value`, НЕ дефолтом
    функции. `tools/check_physics_weathergft.py` намеренно передаёт легаси
    `--coriolis-value 7.29e-5` (= Ω, paper-вариант, без множителя 2 и sin) для
    воспроизведения старых экспериментов — там method_name/tags честно содержат
    `constOmega_legacy`, так что в Comet это не перепутать с каноничным.
    Каноничный 2Ω·sin(45°) — это дефолт здесь и в `utils.physics.Grid.f_constant`.
    """
    return torch.tensor(value)


def coriolis_beta_plane(
    geom: GeometryCPU, f0: float = _F_MID_LATITUDE, beta: float = 1.6e-11
) -> torch.Tensor:
    """f = f0 + β·R·φ, где f0 = 2Ω·sin(45°).

    NB: дефолт f0 изменён с легаси 7.29e-5 на 2Ω·sin(45°). См. coriolis_constant.
    """
    y = geom.radius * geom.latitudes
    return (f0 + beta * y).reshape(1, 1, -1, 1)


def coriolis_spherical(geom: GeometryCPU, omega: float = _OMEGA_EARTH) -> torch.Tensor:
    """f = 2Ω·sin(φ) — полное сферическое приближение."""
    return (2 * omega * torch.sin(geom.latitudes)).reshape(1, 1, -1, 1)


# =============================================================================
# Comet ML logger
# =============================================================================


class Comet72hLogger:
    """Обёртка над comet_ml.Experiment с per-(variable, plvl, lead_hour) логированием.

    Слим-набор (по запросу пользователя — только weighted_rmse и ACC):
        weighted_rmse/<var>/<plvl>hPa     — 5 prog vars × 13 lvls
        acc/<var>/<plvl>hPa               — spatial-Pearson ACC, 5 × 13
        persistence/surface/<var>         — 4 surface vars (persistence-пол,
            тождествен у всех методов: физика surface не прогнозирует)
        nan_count/<var>                   — счётчик NaN/Inf клеток в state
        frac_ic_blown_up                  — фракция IC, сломанных к моменту h

    step = lead-hour ∈ {0, 1, ..., horizon_hours} (default 48).
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
) -> tuple[float, float]:
    """Lat-weighted RMSE + spatial-anomaly ACC. Слим-набор по запросу пользователя.

    Args:
        pred, truth: одинаковые формы ``(B, H, W)`` или ``(B, 1, H, W)``.
        lat_w: ``(H,)`` — нормированные cos(lat)·H/Σcos.

    Returns:
        (wrmse, acc).

    ACC = lat-weighted **spatial-Pearson** между prediction и truth: аномалия =
    поле − его lat-weighted spatial mean (on-the-fly из truth). Не каноничный
    ECMWF-ACC (нужна 30-летняя climatology), но валидно для коротких rollout’ов.
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
        truth = truth.squeeze(1)
    assert pred.dim() == 3, f"Expected (B, H, W), got {pred.shape}"
    _, H, W = pred.shape
    w = lat_w.view(1, H, 1)
    w_sum = float(lat_w.sum().item()) * W

    diff = pred - truth
    wmse_sum = (w * diff * diff).sum(dim=(-1, -2))
    wrmse = torch.sqrt((wmse_sum / w_sum).mean()).item()

    p_mean = (w * pred).sum(dim=(-1, -2), keepdim=True) / w_sum
    t_mean = (w * truth).sum(dim=(-1, -2), keepdim=True) / w_sum
    p_anom = pred - p_mean
    t_anom = truth - t_mean
    num = (w * p_anom * t_anom).sum(dim=(-1, -2))
    den = torch.sqrt(
        (w * p_anom * p_anom).sum(dim=(-1, -2)) * (w * t_anom * t_anom).sum(dim=(-1, -2)) + 1e-30
    )
    acc = (num / den).mean().item()

    return wrmse, acc


def compute_forecast_metrics(
    pred_state: dict[str, torch.Tensor],
    truth_state: dict[str, torch.Tensor],
    geom: GeometryCPU,
) -> dict[str, float]:
    """Слим набор: только lat-weighted RMSE и spatial-anomaly ACC.

    Metric naming (Comet groups by `/`):
        persistence/surface/<var>            — 4 surface vars: физика их НЕ
            прогнозирует (passthrough IC), поэтому метрика тождественна у
            ВСЕХ методов = persistence-ошибка данных. Префикс `persistence/`
            (не `weighted_rmse/`) явно сигналит, что это baseline-пол, а не
            метод-различающая метрика — иначе совпадающие кривные читаются
            как «баг-дубликат» (см. experiments/README.md).
        weighted_rmse/<var>/<plvl>hPa        — 5 prog vars × 13 levels = 65
        acc/<var>/<plvl>hPa                  — 65 (метод-различающие)

    Удалено vs прежней версии (по запросу пользователя):
        weighted_mae/*, weighted_bias/*, rmse/*, mae/*, psnr/*, weighted_bias/surface/*.
    """
    metrics: dict[str, float] = {}
    lat_w = geom.lat_weights

    for var in SURFACE_VARS:
        wrmse, _ = _weighted_stats(pred_state[var], truth_state[var], lat_w)
        metrics[f"persistence/surface/{var}"] = wrmse

    for var in PROGNOSTIC_VARS:
        p_full = pred_state[var]
        t_full = truth_state[var]
        for lvl_idx, plvl in enumerate(PRESSURE_LEVELS_HPA):
            wrmse, acc = _weighted_stats(p_full[:, lvl_idx], t_full[:, lvl_idx], lat_w)
            metrics[f"weighted_rmse/{var}/{plvl}hPa"] = wrmse
            metrics[f"acc/{var}/{plvl}hPa"] = acc

    return metrics


def count_nans_per_var(state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Количество NaN/Inf на каждый prognostic var. q теперь включён (раньше пропускался)."""
    out: dict[str, float] = {}
    for var in PROGNOSTIC_VARS:  # ("z", "t", "u", "v", "q") — теперь с q
        s = state[var]
        out[f"nan_count/{var}"] = float(torch.isnan(s).sum().item() + torch.isinf(s).sum().item())
    return out


# =============================================================================
# Generic rollout driver (default 48h; раньше 72h)
# =============================================================================


def run_72h_rollout(
    *,
    method_name: str,
    rollout_step_fn: Callable[
        [dict[str, torch.Tensor]],
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ],
    geom: GeometryCPU,
    initial_conditions: list[pd.Timestamp],
    memmap_path: str,
    memmap_meta_path: str | None,
    mean_std_path: str,
    horizon_hours: int = 48,
    block_dt_seconds: float = 300.0,
    project_name: str = "WeatherPredictions",
    workspace: str | None = None,
    api_key: str | None = None,
    tags: list[str] | None = None,
    offline: bool = False,
    prepare_hook: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    # Параметры ниже остались только для backward-compat сигнатуры — больше не
    # используются (physics-метрики удалены по запросу пользователя).
    d_x_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    d_y_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    f_field: torch.Tensor | None = None,
) -> None:
    """Прогнать N-часовой rollout по всем IC, логируя в Comet ML.

    Слим-набор метрик: `weighted_rmse/{var}/{plvl}hPa`, `acc/{var}/{plvl}hPa`,
    `persistence/surface/{var}` (пол, тождествен у всех), `nan_count/{var}`,
    `frac_ic_blown_up`.

    Args:
        method_name: human-readable id метода.
        rollout_step_fn: функция (state-dict) → (state_next, rhs_dict). Возвращает
            tuple для backward-compat; второй элемент игнорируется (physics-метрики
            убраны).
        geom: геометрия (для lat_weights).
        initial_conditions: список pd.Timestamp.
        memmap_path: путь к ERA5 memmap.
        memmap_meta_path: meta.json (если None — derived из memmap_path).
        mean_std_path: пустая строка → memmap считается raw физическими единицами.
        horizon_hours: горизонт прогноза (default 48).
        block_dt_seconds: substep, 3600 / block_dt_seconds — substeps per hour.
        project_name, workspace, api_key: Comet creds.
        tags: тэги Comet.
        offline: писать OfflineExperiment (нужен upload_experiment позже).
        prepare_hook: опц. трансформер стартового state после `_prepare_state`
            (E4: балансировка IC). Применяется только к IC, не к truth.
        d_x_fn, d_y_fn, f_field: устарели, игнорируются (оставлены для совместимости).
    """
    del d_x_fn, d_y_fn, f_field  # больше не используются

    if 3600 % int(block_dt_seconds) != 0:
        raise ValueError(f"3600 must be divisible by block_dt={block_dt_seconds}")
    substeps_per_hour = int(3600 // int(block_dt_seconds))

    print(f"[init] Comet experiment for method={method_name}")
    api_key_eff = None if offline else (api_key or os.environ.get("COMET_API_KEY"))
    workspace_eff = workspace or os.environ.get("COMET_WORKSPACE")
    project_name_eff = project_name or os.environ.get("COMET_PROJECT_NAME") or "WeatherPredictions"

    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{method_name}_{horizon_hours}h_{timestamp_str}"
    logger = Comet72hLogger(
        project_name=project_name_eff,
        workspace=workspace_eff,
        api_key=api_key_eff,
        run_name=run_name,
        offline_dir=f"logs/comet_offline/{method_name}",
        tags=(tags or []) + [method_name, f"H{geom.H}", f"W{geom.W}", "cpu", f"{horizon_hours}h"],
    )
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

    print(f"[init] Opening memmap {memmap_path}")
    handle = open_memmap(memmap_path, memmap_meta_path)
    print(f"[init] Memmap shape={handle.shape}, years={handle.meta.get('years')}")
    mean, std = load_mean_std(mean_std_path)

    # БАГ-2 fix: усреднение по ФИНИТНЫМ IC (а не «poison-all», который выбрасывал
    # сигнал выживших траекторий, как только 1/12 IC давал NaN). Сигнал
    # нестабильности несёт отдельная метрика frac_ic_blown_up.
    sum_metrics: dict[int, dict[str, float]] = {h: {} for h in range(horizon_hours + 1)}
    cnt_metrics: dict[int, dict[str, int]] = {h: {} for h in range(horizon_hours + 1)}
    seen_metrics: dict[int, set[str]] = {h: set() for h in range(horizon_hours + 1)}
    nan_ic_count: dict[int, int] = {h: 0 for h in range(horizon_hours + 1)}

    def _accumulate(h: int, m: dict[str, float]) -> None:
        bucket_s = sum_metrics[h]
        bucket_c = cnt_metrics[h]
        bucket_seen = seen_metrics[h]
        for k, v in m.items():
            if v is None:
                continue
            bucket_seen.add(k)  # метрика встречалась (даже если на этом IC NaN)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue  # пропускаем только этот IC, не отравляем остальные
            bucket_s[k] = bucket_s.get(k, 0.0) + float(v)
            bucket_c[k] = bucket_c.get(k, 0) + 1

    t_start = time.time()
    for ic_idx, ts0 in enumerate(initial_conditions):
        print(f"\n[IC {ic_idx + 1}/{len(initial_conditions)}] {ts0}")
        x0 = load_snapshot(handle, ts0, mean, std)
        state = _prepare_state(x0, geom)
        if prepare_hook is not None:
            # E4: балансировка IC (DFI/geostrophic). Только стартовое
            # состояние — truth_state остаётся сырым ERA5 для честных метрик.
            state = prepare_hook(state)

        # h=0: forecast == truth (identity), wrmse=0, acc=1.
        _accumulate(0, compute_forecast_metrics(state, state, geom))
        nan_m0 = count_nans_per_var(state)
        _accumulate(0, nan_m0)

        # БАГ-3 fix: IC, мёртвая уже на инициализации (битая строка memmap,
        # деление в Magnus при p≈0.378·e_s), должна попасть в frac_ic_blown_up@h0.
        ic_blew_up_at: int | None = None
        if any(v > 0 for v in nan_m0.values()):
            ic_blew_up_at = 0
            nan_ic_count[0] += 1
        for hour in range(1, horizon_hours + 1):
            for _ in range(substeps_per_hour):
                state, _ = rollout_step_fn(state)

            ts_truth = ts0 + pd.Timedelta(hours=hour)
            x_truth = load_snapshot(handle, ts_truth, mean, std)
            truth_state = _prepare_state(x_truth, geom)

            forecast_m = compute_forecast_metrics(state, truth_state, geom)
            nan_m = count_nans_per_var(state)
            _accumulate(hour, forecast_m)
            _accumulate(hour, nan_m)

            if ic_blew_up_at is None and any(v > 0 for v in nan_m.values()):
                ic_blew_up_at = hour
                nan_ic_count[hour] += 1

            if hour % 12 == 0 or hour in (1, 6, 24, 48):
                elapsed = time.time() - t_start
                u_blow = state["u"].abs().max().item()
                wrmse_u500 = forecast_m.get("weighted_rmse/u/500hPa", float("nan"))
                print(
                    f"  h={hour:3d}  |u|max={u_blow:.2e}  "
                    f"wrmse(u@500)={wrmse_u500:.3e}  elapsed={elapsed:.0f}s"
                )

    print("\n[log] Pushing per-step metrics to Comet (mean over finite IC)…")
    n_ic = len(initial_conditions)
    for h in range(horizon_hours + 1):
        agg: dict[str, float] = {}
        for k in seen_metrics[h]:
            c = cnt_metrics[h].get(k, 0)
            # Среднее по финитным IC; NaN только если ВСЕ IC дали non-finite
            # (тогда метрика честно неопределена).
            agg[k] = sum_metrics[h][k] / c if c > 0 else float("nan")
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
    q = relhum_to_specific(parts["r"], parts["t"], geom.pressure_pa_t)
    parts["q"] = q
    return parts


_PROG = ("u", "v", "t", "q", "z")


def _lanczos_lowpass(n: int) -> list[float]:
    """Lanczos-оконный идеальный low-pass на 2n+1 отсчётах (k=−n..n).

    Cutoff θ_c = π/n → период отсечки = 2·n·Δt = полный размах фильтра:
    медленные (Россби) моды проходят, быстрые (гравитационные) гасятся.
    Нормирован Σ h_k = 1. Возвращает список длины 2n+1, индекс i ↔ k=i−n.
    """
    coeffs: list[float] = []
    for k in range(-n, n + 1):
        if k == 0:
            h = 1.0 / n  # θ_c/π при θ_c=π/n
        else:
            ideal = math.sin(math.pi * k / n) / (math.pi * k)
            window = math.sin(math.pi * k / n) / (math.pi * k / n)  # Lanczos
            h = ideal * window
        coeffs.append(h)
    s = sum(coeffs)
    return [c / s for c in coeffs]


def _dfi_balance(
    s0: dict[str, torch.Tensor], kernel: PurePDEKernel, span_hours: float
) -> dict[str, torch.Tensor]:
    """Forward-only DFI на СТАБИЛИЗИРОВАННОМ kernel.step + Lanczos low-pass.

    Явный backward-Эйлер на сырой (неустойчивой) физике расходится за
    несколько шагов, поэтому DFI интегрирует ВПЕРЁД через ``kernel.step``
    (на E4 это уже ssp_rk3 + ∇⁴ + polar из E1–E3 — устойчиво). Окно
    [0, 2n]·dt, симметричный Lanczos-low-pass с пиком в n → сбалансированное
    состояние валидно ≈ на n·dt (малый сдвиг ≪ 48 ч; стандартное
    приближение forward-only DFI). Убирает быстрые по времени
    (гравитационные) моды, сохраняет медленный (Россби) поток.
    """
    n = max(1, round(span_hours * 3600.0 / kernel.block_dt))
    h = _lanczos_lowpass(n)  # длина 2n+1, пик в индексе n
    cur = {k: s0[k] for k in _PROG}
    acc = {k: h[0] * cur[k] for k in _PROG}
    with torch.no_grad():
        for i in range(1, 2 * n + 1):
            out = kernel.step(cur["u"], cur["v"], cur["t"], cur["q"], cur["z"])
            cur = {k: out[k] for k in _PROG}
            acc = {k: acc[k] + h[i] * cur[k] for k in _PROG}
    return acc


def _geostrophic_balance(
    s0: dict[str, torch.Tensor], kernel: PurePDEKernel
) -> dict[str, torch.Tensor]:
    """Геострофическая инициализация: ветер из массы (u_g,v_g) = (−z_y,z_x)/f.

    Убирает агеострофический дисбаланс конструктивно. У экватора |f|
    ограничивается снизу (2Ω·sin5°), чтобы не делить на ~0.
    """
    z = s0["z"]
    z_x = kernel.diff.d_x(z)
    z_y = kernel.diff.d_y(z)
    f = kernel.f_field
    f_min = 2.0 * 7.2921e-5 * math.sin(math.radians(5.0))
    sign = torch.sign(f)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    f_safe = torch.where(f.abs() < f_min, sign * f_min, f)
    out = dict(s0)
    out["u"] = -z_y / f_safe
    out["v"] = z_x / f_safe
    return out


def balance_initial_state(
    state: dict[str, torch.Tensor],
    kernel: PurePDEKernel,
    mode: str,
    span_hours: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Сбалансировать IC до rollout (E4): убрать initialization-shock.

    Args:
        state: dict с прогностическими (u,v,t,q,z) + surface/r полями.
        kernel: PurePDEKernel (нужны `.step`, `.block_dt`, `.f_field`, `.diff`).
        mode: ``none`` (passthrough) | ``dfi`` | ``geostrophic``.
        span_hours: полу-размах DFI-окна в часах (forward-only, см.
            :func:`_dfi_balance`). Не используется для ``geostrophic``.

    Returns:
        Новый dict: прогностические заменены сбалансированными, остальные
        (t2m/u10/v10/tp/r) — passthrough. LBYL-guard: если балансировка
        дала NaN/Inf — возвращается исходный state (с предупреждением).
    """
    if mode == "none":
        return state
    s0 = {k: state[k] for k in _PROG}
    if mode == "dfi":
        bal = _dfi_balance(s0, kernel, span_hours)
    elif mode == "geostrophic":
        bal = _geostrophic_balance(s0, kernel)
    else:
        raise ValueError(f"Unknown balance-ic mode {mode!r}")
    if not all(bool(torch.isfinite(v).all()) for v in bal.values()):
        print(f"[balance_ic] mode={mode} дал NaN/Inf → fallback на сырой IC")
        return state
    out = dict(state)
    out.update(bal)
    return out


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
    "adiabatic_temperature_tendency",
    "GeometryCPU",
    "coriolis_constant",
    "coriolis_beta_plane",
    "coriolis_spherical",
    "Comet72hLogger",
    "compute_forecast_metrics",
    "count_nans_per_var",
    "balance_initial_state",
    "run_72h_rollout",
]
