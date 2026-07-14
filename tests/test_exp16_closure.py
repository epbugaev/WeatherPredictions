"""Тесты статистики предпроверки замыкания (exp16, армы X1/X2).

Всё — на аналитически известных ответах: корреляция поля с самим собой равна 1, R²
точной линейной комбинации предикторов равен 1, контраст режима у сдвинутого поля
равен сдвигу. Диагностика, на которой строится предсказание для X1, обязана быть
проверяемой сама по себе — иначе мы снова будем читать шум как сигнал.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_LIB = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/16_model_ablation_ladder/closure/closure_lib.py"
)
_spec = importlib.util.spec_from_file_location("exp16_closure_lib", _LIB)
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)


def test_pearson_of_a_field_with_itself_is_one() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(4, 8, 16))
    assert cl.spatial_pearson(field, field) == pytest.approx(np.ones(4))


def test_pearson_flips_sign_with_the_field() -> None:
    rng = np.random.default_rng(1)
    field = rng.normal(size=(3, 8, 16))
    assert cl.spatial_pearson(field, -field) == pytest.approx(-np.ones(3))


def test_pearson_is_invariant_to_scale_and_offset() -> None:
    """Единицы Q_θ (нормированные σ) и cond ((кг/кг)/с) несопоставимы — и не должны."""
    rng = np.random.default_rng(2)
    field = rng.normal(size=(2, 8, 16))
    rescaled = 137.0 * field - 4.2
    assert cl.spatial_pearson(field, rescaled) == pytest.approx(np.ones(2))


def test_pearson_is_nan_on_a_constant_field() -> None:
    """cond вне зоны насыщения тождественно нулевой — корреляция не определена."""
    rng = np.random.default_rng(3)
    result = cl.spatial_pearson(np.zeros((2, 4, 4)), rng.normal(size=(2, 4, 4)))
    assert np.isnan(result).all()


def test_explained_variance_is_one_for_an_exact_linear_combination() -> None:
    rng = np.random.default_rng(4)
    predictors = rng.normal(size=(5, 3, 8, 16))
    target = 2.0 * predictors[:, 0] - 0.5 * predictors[:, 1] + 7.0
    assert cl.explained_variance(target, predictors) == pytest.approx(np.ones(5), abs=1e-6)


def test_explained_variance_is_near_zero_for_independent_noise() -> None:
    rng = np.random.default_rng(5)
    predictors = rng.normal(size=(6, 2, 16, 32))
    target = rng.normal(size=(6, 16, 32))
    assert cl.explained_variance(target, predictors).max() < 0.05


def test_explained_variance_is_nan_for_a_constant_target() -> None:
    rng = np.random.default_rng(6)
    result = cl.explained_variance(np.full((2, 4, 4), 3.0), rng.normal(size=(2, 2, 4, 4)))
    assert np.isnan(result).all()


def test_regime_contrast_recovers_a_known_offset() -> None:
    """Поле, поднятое на +5 внутри режима, обязано дать контраст ровно +5."""
    rng = np.random.default_rng(7)
    field = rng.normal(size=(4, 8, 16))
    mask = np.zeros_like(field, dtype=bool)
    mask[:, :, :4] = True
    field = field + 5.0 * mask

    result = cl.regime_contrast(field, mask)
    assert result["contrast"] == pytest.approx(5.0, abs=0.3)
    assert result["mask_fraction"] == pytest.approx(0.25)


def test_regime_contrast_handles_an_empty_regime() -> None:
    """На уровнях без насыщения режим пуст — не падаем, отдаём NaN."""
    result = cl.regime_contrast(np.ones((2, 4, 4)), np.zeros((2, 4, 4), dtype=bool))
    assert np.isnan(result["inside"])
    assert result["mask_fraction"] == 0.0


def test_summarize_ignores_nans_and_counts_strong_correlations() -> None:
    values = np.array([0.9, -0.8, 0.1, np.nan, 0.5, -0.4])
    summary = cl.summarize(values)
    assert summary["frac_pos"] == pytest.approx(2 / 5)  # 0.9, 0.5 > +0.3
    assert summary["frac_neg"] == pytest.approx(2 / 5)  # -0.8, -0.4 < -0.3
    assert summary["frac_valid"] == pytest.approx(5 / 6)


def test_summarize_on_all_nan_does_not_crash() -> None:
    summary = cl.summarize(np.full(4, np.nan))
    assert np.isnan(summary["mean"])
    assert summary["frac_valid"] == 0.0
