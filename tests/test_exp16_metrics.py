"""Пины библиотеки метрик exp16 (`metrics/metrics_lib.py`).

Каждая метрика проверяется на аналитически известном ответе, а не на «не упало».
Отдельно закреплена геометрия: широты USA-кропа 16.875..63.28125°N, ряды юг→север
(аудит exp14) — `utils/metrics.py` считает их как «полюс→полюс» и потому здесь не
используется.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "docs/experiments/16_model_ablation_ladder/metrics"

spec = importlib.util.spec_from_file_location("exp16_metrics_lib", EXP_DIR / "metrics_lib.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


# --- геометрия -----------------------------------------------------------------


def test_crop_latitudes_run_south_to_north() -> None:
    lats = ml.crop_latitudes_deg(32)
    assert lats.shape == (32,)
    assert math.isclose(float(lats[0]), 16.875, abs_tol=1e-6)  # юг
    assert math.isclose(float(lats[-1]), 63.28125, abs_tol=1e-6)  # север
    assert bool((np.diff(lats) > 0).all())


def test_lat_weights_are_cosine_and_mean_one() -> None:
    weights = ml.lat_weights(32)
    lats = ml.crop_latitudes_deg(32)
    expected = np.cos(np.deg2rad(lats))
    assert np.allclose(weights.numpy(), expected / expected.mean(), atol=1e-6)
    # южный край кропа весит больше северного (cos убывает к полюсу)
    assert weights[0] > weights[-1]


# --- RMSE / bias / std / ACC ---------------------------------------------------


def test_weighted_rmse_constant_error() -> None:
    weights = ml.lat_weights(4)
    obs = torch.zeros(2, 3, 5, 4, 6)
    pred = torch.full_like(obs, 2.0)
    rmse = ml.weighted_rmse(pred, obs, weights)
    assert rmse.shape == (2, 3, 5)
    assert torch.allclose(rmse, torch.full_like(rmse, 2.0), atol=1e-5)


def test_weighted_bias_signed() -> None:
    weights = ml.lat_weights(4)
    obs = torch.zeros(1, 1, 1, 4, 6)
    pred = torch.full_like(obs, -3.0)
    bias = ml.weighted_bias(pred, obs, weights)
    assert torch.allclose(bias, torch.full_like(bias, -3.0), atol=1e-5)


def test_weighted_spatial_std_of_constant_field_is_zero() -> None:
    weights = ml.lat_weights(4)
    field = torch.full((1, 1, 1, 4, 6), 7.0)
    assert torch.allclose(ml.weighted_std(field, weights), torch.zeros(1, 1, 1), atol=1e-6)


def test_weighted_acc_perfect_and_anticorrelated() -> None:
    weights = ml.lat_weights(4)
    anomaly = torch.randn(2, 1, 1, 4, 6)
    perfect = ml.weighted_acc(anomaly, anomaly, weights)
    assert torch.allclose(perfect, torch.ones_like(perfect), atol=1e-5)
    flipped = ml.weighted_acc(-anomaly, anomaly, weights)
    assert torch.allclose(flipped, -torch.ones_like(flipped), atol=1e-5)


# --- CSI ------------------------------------------------------------------------


def test_contingency_counts_and_csi() -> None:
    # ряд из 4 ячеек: obs событие в 0,1; pred событие в 1,2 -> TP=1, FP=1, FN=1
    weights = torch.ones(1)
    obs = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]])
    pred = torch.tensor([[[[[0.0, 1.0, 1.0, 0.0]]]]])
    tp, fp, fn = ml.contingency_counts(pred, obs, torch.tensor([0.5]), weights)
    assert tp.shape == (1, 1, 1, 1)
    assert float(tp) == 1.0 and float(fp) == 1.0 and float(fn) == 1.0
    csi = ml.csi_from_counts(tp, fp, fn)
    assert math.isclose(float(csi), 1.0 / 3.0, rel_tol=1e-6)  # TP/(TP+FP+FN)


def test_csi_is_one_for_perfect_forecast() -> None:
    weights = torch.ones(2)
    obs = torch.tensor([[[[[0.0, 2.0], [2.0, 0.0]]]]])
    tp, fp, fn = ml.contingency_counts(obs, obs, torch.tensor([1.0]), weights)
    assert float(ml.csi_from_counts(tp, fp, fn)) == 1.0


# --- FSS ------------------------------------------------------------------------


def test_fss_is_one_for_perfect_forecast() -> None:
    obs = torch.tensor([[[[[0.0, 2.0, 0.0, 2.0], [2.0, 0.0, 2.0, 0.0]]]]])
    num, den = ml.fss_terms(obs, obs, torch.tensor([1.0]), neighborhood=3)
    assert float(num) == 0.0  # числитель = MSE долей = 0
    assert float(ml.fss_from_terms(num, den)) == 1.0


def test_fss_penalises_displacement_less_than_csi() -> None:
    # сдвиг события на 1 ячейку: CSI = 0 (нет пересечения), FSS с окном 3 > 0
    obs = torch.zeros(1, 1, 1, 1, 7)
    obs[..., 2] = 5.0
    pred = torch.zeros_like(obs)
    pred[..., 3] = 5.0
    thresholds = torch.tensor([1.0])
    tp, fp, fn = ml.contingency_counts(pred, obs, thresholds, torch.ones(1))
    assert float(ml.csi_from_counts(tp, fp, fn)) == 0.0
    num, den = ml.fss_terms(pred, obs, thresholds, neighborhood=3)
    assert float(ml.fss_from_terms(num, den)) > 0.0


# --- W1, PSD --------------------------------------------------------------------


def test_wasserstein1_shift_equals_shift() -> None:
    # W1 между распределением и им же, сдвинутым на c, равно |c|
    obs = torch.randn(1, 1, 1, 4, 8)
    pred = obs + 3.0
    w1 = ml.wasserstein1(pred, obs)
    assert w1.shape == (1, 1, 1)
    assert torch.allclose(w1, torch.full_like(w1, 3.0), atol=1e-5)


def test_wasserstein1_is_zero_for_permuted_field() -> None:
    # W1 сравнивает РАСПРЕДЕЛЕНИЯ: перестановка тех же значений даёт 0
    obs = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    pred = torch.tensor([[[[[4.0, 3.0], [2.0, 1.0]]]]])
    assert float(ml.wasserstein1(pred, obs)) == 0.0


def test_zonal_psd_finds_planted_wavenumber() -> None:
    # поле cos(2*2pi*x/W) -> вся мощность на зональном волновом числе m=2
    n_lon = 64
    x = torch.arange(n_lon, dtype=torch.float32)
    field = torch.cos(2 * 2 * math.pi * x / n_lon).view(1, 1, 1, 1, n_lon).repeat(1, 1, 1, 4, 1)
    psd = ml.zonal_psd(field, ml.lat_weights(4))
    assert psd.shape == (1, 1, 1, n_lon // 2 + 1)
    assert int(psd.argmax()) == 2


# --- бутстрап -------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    per_sample = rng.normal(loc=5.0, scale=1.0, size=(500, 2, 3))
    mean, std, low, high = ml.bootstrap_ci(per_sample, n_resamples=200, seed=0)
    assert mean.shape == (2, 3) and std.shape == (2, 3)
    assert np.all(low < mean) and np.all(mean < high)
    assert np.allclose(mean, 5.0, atol=0.2)


def test_bootstrap_ci_ignores_nan_samples() -> None:
    per_sample = np.full((10, 1), 4.0)
    per_sample[3] = np.nan  # сэмпл без событий (CSI не определён) не должен тянуть среднее
    mean, _, _, _ = ml.bootstrap_ci(per_sample, n_resamples=50, seed=0)
    assert np.allclose(mean, 4.0)


def test_bootstrap_ratio_ci_aggregates_sums_not_means() -> None:
    # CSI/FSS агрегируются как Σчислитель/Σзнаменатель. Проверка: сэмпл без событий
    # (0/0) не портит агрегат, а среднее по-сэмпльных отношений — испортило бы.
    num = np.array([[1.0], [3.0], [0.0]])
    den = np.array([[2.0], [6.0], [0.0]])
    mean, _, low, high = ml.bootstrap_ratio_ci(num, den, n_resamples=100, seed=0)
    assert np.allclose(mean, 0.5)  # (1+3+0) / (2+6+0)
    assert np.all(low <= mean) and np.all(mean <= high)


def test_paired_delta_ci_is_tighter_than_independent_cis() -> None:
    # Армы считаются на ОДНИХ И ТЕХ ЖЕ сэмплах: «трудные дни» поднимают RMSE обоим.
    # Парный бутстрап это сокращает, независимые CI — нет. Проверяем, что парный CI
    # разницы существенно уже, чем разброс самих метрик.
    rng = np.random.default_rng(0)
    difficulty = rng.normal(50.0, 15.0, size=(400, 1))  # общая «трудность» сэмпла
    base = difficulty + rng.normal(0.0, 0.5, size=(400, 1))
    arm = 0.97 * difficulty + rng.normal(0.0, 0.5, size=(400, 1))  # арм стабильно на 3% лучше
    mean, _, low, high = ml.bootstrap_paired_delta_ci(arm, base, n_resamples=300, seed=0)
    assert np.allclose(mean, -3.0, atol=0.3)  # Δ% ≈ −3
    assert high < 0.0  # разница значима: интервал не накрывает ноль
    assert (high - low) < 1.0  # и он узкий, хотя std самих метрик огромен
    assert np.std(base) > 10.0


def test_paired_ci_of_arbitrary_statistic() -> None:
    # Общий парный бутстрап: статистика от ОБЩЕГО набора индексов. Проверяем на
    # абсолютной разнице ограниченной оценки (ACC/CSI/FSS так и надо сравнивать).
    rng = np.random.default_rng(1)
    shared = rng.normal(0.0, 0.2, size=(300, 1))
    base = 0.60 + shared
    arm = 0.65 + shared  # стабильно на 0.05 выше
    stat = lambda idx: 100.0 * (arm[idx].mean(axis=0) - base[idx].mean(axis=0))  # noqa: E731
    mean, _, low, high = ml.bootstrap_paired_statistic_ci(stat, 300, n_resamples=300, seed=0)
    assert np.allclose(mean, 5.0, atol=0.5)  # +5 пунктов
    assert low > 0.0  # значимо, несмотря на большой общий разброс


def test_relative_delta_would_explode_where_absolute_is_fine() -> None:
    # Причина бага: у CSI знаменатель бывает нулём (нет попаданий) -> inf.
    # Абсолютная разница на тех же данных определена.
    base = np.array([[0.0], [0.0], [0.0]])
    arm = np.array([[0.2], [0.1], [0.0]])
    rel, _, _, _ = ml.bootstrap_paired_delta_ci(arm, base, n_resamples=20, seed=0)
    assert not np.isfinite(rel).all()  # относительная — не определена
    stat = lambda idx: 100.0 * (arm[idx].mean(axis=0) - base[idx].mean(axis=0))  # noqa: E731
    absolute, _, _, _ = ml.bootstrap_paired_statistic_ci(stat, 3, n_resamples=20, seed=0)
    assert np.isfinite(absolute).all()
