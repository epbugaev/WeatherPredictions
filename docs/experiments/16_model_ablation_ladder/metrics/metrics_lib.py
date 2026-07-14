"""Библиотека метрик прогноза для лестницы exp16 (самодостаточна).

Почему не `utils/metrics.py`: тамошняя `lat()` переводит индекс ряда в широту как
``90 − j·180/(N−1)``, то есть считает, что тензор покрывает **весь глобус**. Наш
домен — USA-кроп: 32 ряда с 16.875°N по 63.28125°N, ряды идут **юг→север** (геометрия
выверена в аудите exp14, см. `18_level_resolved_physics/level_diagnostics.py`). На
кропе та формула зануляет веса южного и северного краёв и переоценивает середину.
Здесь широты берутся настоящие; `utils/metrics.py` не трогается — он в train-цикле,
и его правка сделала бы val-метрики несравнимыми с прошлыми ранами.

Соглашения:
  * поля — ``(B, T, C, H, W)`` в физических единицах, ряды юг→север;
  * веса широт — ``cos(lat)``, нормированы на среднее 1 (масштаб метрик сохраняется);
  * метрики возвращают ``(B, T, C)`` (или ``(B, T, C, K)`` при порогах), чтобы
    бутстрап по сэмплам делался снаружи.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# USA-кроп v4: строки 75..106 глобальной сетки 128×256 (1.40625°), юг→север.
CROP_LAT_RANGE_DEG: tuple[float, float] = (16.875, 63.28125)
CROP_LON_STEP_DEG: float = 1.40625
SPATIAL_DIMS = (-2, -1)


def crop_latitudes_deg(n_rows: int = 32) -> np.ndarray:
    """Широты рядов USA-кропа в градусах, юг→север.

    Args:
        n_rows: число рядов по широте (32 для v4-кропа).

    Returns:
        ``np.ndarray`` формы ``(n_rows,)`` — от 16.875°N до 63.28125°N.
    """
    return np.linspace(CROP_LAT_RANGE_DEG[0], CROP_LAT_RANGE_DEG[1], n_rows)


def lat_weights(n_rows: int = 32, device: torch.device | None = None) -> torch.Tensor:
    """Веса широт ``cos(lat)``, нормированные на среднее 1.

    Args:
        n_rows: число рядов по широте.
        device: устройство результата.

    Returns:
        ``torch.Tensor`` формы ``(n_rows,)``.
    """
    weights = np.cos(np.deg2rad(crop_latitudes_deg(n_rows)))
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _broadcast_weights(weights: torch.Tensor) -> torch.Tensor:
    """Веса ``(H,)`` → форма для broadcast на ``(B, T, C, H, W)``."""
    return weights.view(1, 1, 1, -1, 1)


def _weighted_mean(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Средневзвешенное по (H, W): ``(B, T, C, H, W)`` → ``(B, T, C)``."""
    w = _broadcast_weights(weights)
    return (field * w).sum(dim=SPATIAL_DIMS) / (w.expand_as(field).sum(dim=SPATIAL_DIMS))


def weighted_rmse(pred: torch.Tensor, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Lat-weighted RMSE по каждому (сэмпл, шаг, канал).

    Args:
        pred: прогноз ``(B, T, C, H, W)`` в физических единицах.
        obs: истина той же формы.
        weights: веса широт ``(H,)`` из `lat_weights`.

    Returns:
        ``(B, T, C)``.
    """
    return torch.sqrt(_weighted_mean((pred - obs) ** 2, weights))


def weighted_bias(pred: torch.Tensor, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Lat-weighted систематическая ошибка ``mean(pred − obs)`` → ``(B, T, C)``."""
    return _weighted_mean(pred - obs, weights)


def weighted_std(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Lat-weighted пространственный std поля → ``(B, T, C)``.

    Отношение ``weighted_std(pred) / weighted_std(obs)`` показывает, не сглаживает
    ли модель поле (у авторегрессии дисперсия падает с ростом лида).
    """
    mean = _weighted_mean(field, weights).unsqueeze(-1).unsqueeze(-1)
    return torch.sqrt(_weighted_mean((field - mean) ** 2, weights))


def weighted_acc(
    pred_anomaly: torch.Tensor, obs_anomaly: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Lat-weighted ACC (anomaly correlation) по (сэмпл, шаг, канал).

    Args:
        pred_anomaly: аномалия прогноза (поле минус климатология), ``(B, T, C, H, W)``.
        obs_anomaly: аномалия истины той же формы.
        weights: веса широт ``(H,)``.

    Returns:
        ``(B, T, C)`` в диапазоне ``[-1, 1]``.
    """
    w = _broadcast_weights(weights)
    covariance = (w * pred_anomaly * obs_anomaly).sum(dim=SPATIAL_DIMS)
    pred_power = (w * pred_anomaly**2).sum(dim=SPATIAL_DIMS)
    obs_power = (w * obs_anomaly**2).sum(dim=SPATIAL_DIMS)
    return covariance / torch.sqrt(pred_power * obs_power).clamp_min(1e-12)


def contingency_counts(
    pred: torch.Tensor, obs: torch.Tensor, thresholds: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Взвешенные счётчики попаданий/ложных тревог/пропусков для набора порогов.

    Событие — превышение порога. Счётчики взвешены по широте, поэтому южные ряды
    кропа (более «широкие» ячейки) весят больше — так же, как в RMSE.

    Args:
        pred: прогноз ``(B, T, C, H, W)``.
        obs: истина той же формы.
        thresholds: пороги ``(K,)`` или ``(C, K)`` (свой набор на канал).
        weights: веса широт ``(H,)``.

    Returns:
        Тройка ``(tp, fp, fn)``, каждый ``(B, T, C, K)``.
    """
    w = _broadcast_weights(weights)
    if thresholds.dim() == 1:
        levels = thresholds.view(1, 1, 1, -1, 1, 1)
    else:
        levels = thresholds.view(1, 1, *thresholds.shape, 1, 1)  # (1,1,C,K,1,1)
    pred_event = (pred.unsqueeze(3) > levels).to(pred.dtype)  # (B,T,C,K,H,W)
    obs_event = (obs.unsqueeze(3) > levels).to(obs.dtype)
    cell_weight = w.unsqueeze(3)  # (1,1,1,1,H,1)
    tp = (pred_event * obs_event * cell_weight).sum(dim=SPATIAL_DIMS)
    fp = (pred_event * (1.0 - obs_event) * cell_weight).sum(dim=SPATIAL_DIMS)
    fn = ((1.0 - pred_event) * obs_event * cell_weight).sum(dim=SPATIAL_DIMS)
    return tp, fp, fn


def csi_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """CSI = TP / (TP + FP + FN); при полном отсутствии событий возвращает NaN.

    NaN (а не 0) — чтобы усреднение по сэмплам не считало «нет событий» за провал:
    такие сэмплы должны выпадать из среднего, а не тянуть его вниз.
    """
    denominator = tp + fp + fn
    return torch.where(denominator > 0, tp / denominator.clamp_min(1e-12), torch.nan)


def _fractions(event: torch.Tensor, neighborhood: int) -> torch.Tensor:
    """Доля событий в скользящем окне ``neighborhood × neighborhood``.

    Края дополняются нулями («за границей домена событий нет») — стандартный приём
    для FSS: у прогноза и у истины края разбавляются одинаково, поэтому сравнение
    остаётся честным.
    """
    if neighborhood == 1:
        return event
    batch, time, channels, thresholds, height, width = event.shape
    flat = event.reshape(-1, 1, height, width)
    pad = neighborhood // 2
    flat = F.pad(flat, (pad, pad, pad, pad), mode="constant", value=0.0)
    kernel = torch.ones(1, 1, neighborhood, neighborhood, dtype=event.dtype, device=event.device)
    smoothed = F.conv2d(flat, kernel) / (neighborhood * neighborhood)
    return smoothed.reshape(batch, time, channels, thresholds, height, width)


def fss_terms(
    pred: torch.Tensor, obs: torch.Tensor, thresholds: torch.Tensor, neighborhood: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Числитель и знаменатель Fractions Skill Score (агрегируются по сэмплам ОТДЕЛЬНО).

    FSS = 1 − Σnum / Σden, где num — MSE долей событий в окне, den — референсный
    MSE (сумма мощностей долей). Суммировать надо числитель и знаменатель по
    отдельности, а не усреднять готовый FSS — иначе сэмплы без событий портят среднее.

    Args:
        pred: прогноз ``(B, T, C, H, W)``.
        obs: истина той же формы.
        thresholds: пороги ``(K,)`` или ``(C, K)``.
        neighborhood: сторона окна в ячейках (1 — без окна, тогда FSS ≈ по-ячеечный).

    Returns:
        ``(num, den)``, каждый ``(B, T, C, K)``.
    """
    if thresholds.dim() == 1:
        levels = thresholds.view(1, 1, 1, -1, 1, 1)
    else:
        levels = thresholds.view(1, 1, *thresholds.shape, 1, 1)
    pred_fraction = _fractions((pred.unsqueeze(3) > levels).to(pred.dtype), neighborhood)
    obs_fraction = _fractions((obs.unsqueeze(3) > levels).to(obs.dtype), neighborhood)
    num = ((pred_fraction - obs_fraction) ** 2).mean(dim=SPATIAL_DIMS)
    den = (pred_fraction**2).mean(dim=SPATIAL_DIMS) + (obs_fraction**2).mean(dim=SPATIAL_DIMS)
    return num, den


def fss_from_terms(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """FSS = 1 − num/den; при нулевом знаменателе (нигде нет событий) — NaN."""
    return torch.where(den > 0, 1.0 - num / den.clamp_min(1e-12), torch.nan)


def wasserstein1(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """W1 между эмпирическими распределениями значений поля → ``(B, T, C)``.

    Расстояние Вассерштейна-1 для двух выборок равного размера равно среднему
    ``|sorted(pred) − sorted(obs)|``. Мера **распределения**, а не поточечная:
    перестановка тех же значений даёт 0. Ловит смещение и сжатие гистограммы
    (сглаживание), невидимые для RMSE.
    """
    batch, time, channels = pred.shape[:3]
    flat_pred = pred.reshape(batch, time, channels, -1).sort(dim=-1).values
    flat_obs = obs.reshape(batch, time, channels, -1).sort(dim=-1).values
    return (flat_pred - flat_obs).abs().mean(dim=-1)


def zonal_psd(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Спектр мощности по зональному волновому числу m → ``(B, T, C, W//2 + 1)``.

    Сферические гармоники (степень l) на региональном кропе не определены — домен
    не сфера. По долготе кроп периодичен по построению сетки, поэтому берём FFT
    вдоль долготы: получаем мощность по зональному волновому числу ``m``, усреднённую
    по рядам с весом ``cos(lat)``. Падение мощности прогноза на больших m относительно
    истины = модель сглаживает мелкий масштаб.

    Args:
        field: ``(B, T, C, H, W)``.
        weights: веса широт ``(H,)``.

    Returns:
        ``(B, T, C, W // 2 + 1)`` — мощность по m = 0..W/2.
    """
    spectrum = torch.fft.rfft(field, dim=-1)
    power = (spectrum.real**2 + spectrum.imag**2) / field.shape[-1]
    w = weights.view(1, 1, 1, -1, 1)
    return (power * w).sum(dim=-2) / w.sum()


def channel_quantiles(obs: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Пороги-квантили наблюдённого поля на канал: ``(C, K)``.

    Пороги CSI/FSS берутся из ИСТИНЫ (одни и те же для всех армов), поэтому армы
    сравниваются на одном и том же множестве событий.

    Args:
        obs: истина ``(B, T, C, H, W)`` (обычно вся валидация одним куском).
        quantiles: уровни, напр. ``(0.90, 0.95, 0.99)``.

    Returns:
        ``(C, K)`` — пороги в физических единицах.
    """
    channels = obs.shape[2]
    flat = obs.permute(2, 0, 1, 3, 4).reshape(channels, -1)
    levels = torch.tensor(quantiles, dtype=flat.dtype, device=flat.device)
    return torch.quantile(flat, levels, dim=-1).T.contiguous()


def bootstrap_ci(
    per_sample: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Бутстрап-CI среднего по оси сэмплов (ось 0).

    Args:
        per_sample: ``(S, ...)`` — значения метрики по сэмплам (NaN игнорируются).
        n_resamples: число ресэмплов с возвращением.
        confidence: ширина интервала (0.95 → перцентили 2.5 и 97.5).
        seed: сид генератора (воспроизводимость).

    Returns:
        ``(mean, std, ci_low, ci_high)``, каждый формы ``per_sample.shape[1:]``.
    """
    rng = np.random.default_rng(seed)
    n_samples = per_sample.shape[0]
    mean = np.nanmean(per_sample, axis=0)
    std = np.nanstd(per_sample, axis=0)
    draws = np.empty((n_resamples, *per_sample.shape[1:]), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_samples, n_samples)
        draws[i] = np.nanmean(per_sample[idx], axis=0)
    tail = (1.0 - confidence) / 2.0
    ci_low = np.percentile(draws, 100.0 * tail, axis=0)
    ci_high = np.percentile(draws, 100.0 * (1.0 - tail), axis=0)
    return mean, std, ci_low, ci_high


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Бутстрап-CI для агрегата вида ``Σчислитель / Σзнаменатель`` по оси сэмплов.

    CSI и FSS так и определены: суммировать надо счётчики (TP, FP, FN) или термы
    (num, den), а не усреднять готовые по-сэмпльные значения — сэмпл без событий даёт
    0/0 и, будучи посчитанным как «CSI = 0», занизил бы среднее.

    Args:
        numerator: ``(S, ...)`` — по-сэмпльный числитель (для CSI это TP).
        denominator: ``(S, ...)`` — по-сэмпльный знаменатель (для CSI это TP+FP+FN).
        n_resamples: число ресэмплов с возвращением.
        confidence: ширина интервала.
        seed: сид генератора.

    Returns:
        ``(mean, std, ci_low, ci_high)`` формы ``numerator.shape[1:]``; ``mean`` — это
        агрегат на полной выборке, ``std`` — разброс бутстрап-распределения.
    """
    rng = np.random.default_rng(seed)
    n_samples = numerator.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = numerator.sum(axis=0) / denominator.sum(axis=0)
    draws = np.empty((n_resamples, *numerator.shape[1:]), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_samples, n_samples)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[i] = numerator[idx].sum(axis=0) / denominator[idx].sum(axis=0)
    tail = (1.0 - confidence) / 2.0
    ci_low = np.nanpercentile(draws, 100.0 * tail, axis=0)
    ci_high = np.nanpercentile(draws, 100.0 * (1.0 - tail), axis=0)
    return mean, np.nanstd(draws, axis=0), ci_low, ci_high


def bootstrap_paired_delta_ci(
    arm: np.ndarray,
    base: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Бутстрап-CI относительной разницы ``arm`` и ``base`` в процентах, **парный**.

    Армы прогоняются по одним и тем же валидационным сэмплам, поэтому ресэмплить надо
    **общий** набор индексов: «трудный день» поднимает ошибку у обоих армов, и при
    парном ресэмпле эта общая изменчивость сокращается. Независимые CI каждого арма
    её не убирают и переоценивают неопределённость разницы в разы — по ним нельзя
    судить, отличаются ли армы между собой.

    Args:
        arm: ``(S, ...)`` — по-сэмпльная метрика арма.
        base: ``(S, ...)`` — она же у baseline (те же сэмплы, тот же порядок).
        n_resamples: число ресэмплов с возвращением.
        confidence: ширина интервала.
        seed: сид генератора.

    Returns:
        ``(mean, std, ci_low, ci_high)`` в **процентах** относительно baseline;
        разница значима, если интервал не накрывает 0.
    """
    rng = np.random.default_rng(seed)
    n_samples = arm.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = 100.0 * (np.nanmean(arm, axis=0) / np.nanmean(base, axis=0) - 1.0)
    draws = np.empty((n_resamples, *arm.shape[1:]), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_samples, n_samples)  # ОДИН набор индексов на оба арма
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[i] = 100.0 * (
                np.nanmean(arm[idx], axis=0) / np.nanmean(base[idx], axis=0) - 1.0
            )
    tail = (1.0 - confidence) / 2.0
    return (
        mean,
        np.nanstd(draws, axis=0),
        np.nanpercentile(draws, 100.0 * tail, axis=0),
        np.nanpercentile(draws, 100.0 * (1.0 - tail), axis=0),
    )


def wavelength_km(wavenumber: np.ndarray, mean_latitude_deg: float = 40.0) -> np.ndarray:
    """Длина волны (км) для зонального волнового числа m на широте домена.

    Домен покрывает 90° долготы (64 ячейки × 1.40625°), поэтому m-я гармоника
    укладывается в ``L / m``, где ``L`` — зональная протяжённость кропа.
    """
    earth_radius_km = 6371.0
    domain_width_km = (
        2 * math.pi * earth_radius_km * math.cos(math.radians(mean_latitude_deg)) * (90.0 / 360.0)
    )
    with np.errstate(divide="ignore"):
        return np.where(wavenumber > 0, domain_width_km / np.maximum(wavenumber, 1), np.inf)
