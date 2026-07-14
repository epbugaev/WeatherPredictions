"""Чистая статистика предпроверки «замыкание или свободный источник».

Вопрос: чем на самом деле является Q_θ? Либо **членом замыкания**, который чинит
конкретную дырку в уравнениях ядра (влажностный бюджет односторонний: конденсация —
строго сток, испарения и подсеточной конвекции нет вовсе), либо **свободным
источником**, который выучил испарение сам и в уравнениях не нуждается. От ответа
зависит предсказание для арма X1 (Q_θ без уравнений): в первом случае X1 должен быть
заметно слабее −2.98 %, во втором — примерно равен.

Проверяется на **уже обученном** чекпоинте R3, без единой эпохи обучения: смотрим, с
чем скоррелирован выход Q_θ — с внутренними полями ядра (конденсационный сток `cond`,
вертикальная скорость ω) или со статической географией (орография, широта, суша/море).

Все функции — чистые, без torch и без модели. См. ``closure_probe.py``.
"""

from __future__ import annotations

import numpy as np

RIDGE = 1e-8  # регуляризация нормальных уравнений: гео-предикторы почти коллинеарны


def spatial_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Пространственная корреляция Пирсона по двум последним осям.

    Args:
        a: ``np.ndarray`` формы ``(..., H, W)``.
        b: ``np.ndarray`` той же формы.

    Returns:
        ``np.ndarray`` формы ``(...)`` — коэффициент корреляции; **NaN** там, где хотя
        бы одно поле постоянно (у Q_θ это штатная ситуация: zero-init гасит выход на
        части каналов, а `cond` вне зоны насыщения тождественно нулевой).
    """
    assert a.shape == b.shape, f"формы не совпадают: {a.shape} vs {b.shape}"
    flat_a = a.reshape(*a.shape[:-2], -1)
    flat_b = b.reshape(*b.shape[:-2], -1)
    centered_a = flat_a - flat_a.mean(axis=-1, keepdims=True)
    centered_b = flat_b - flat_b.mean(axis=-1, keepdims=True)
    norm_a = np.sqrt((centered_a**2).sum(axis=-1))
    norm_b = np.sqrt((centered_b**2).sum(axis=-1))
    covariance = (centered_a * centered_b).sum(axis=-1)
    denominator = norm_a * norm_b
    return np.divide(
        covariance,
        denominator,
        out=np.full(covariance.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def explained_variance(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    """Доля дисперсии ``target``, объяснённая линейной моделью на ``predictors``.

    Обычный МНК со свободным членом, отдельно для каждого ведущего индекса. Именно
    эта величина разводит гипотезы: если выход Q_θ на 80 % объясняется географией, то
    «диабатический источник» — эвфемизм для static-geo эмбеддинга, и нужен арм X2.

    Args:
        target: ``np.ndarray`` формы ``(N, H, W)``.
        predictors: ``np.ndarray`` формы ``(N, K, H, W)`` — K полей-регрессоров.

    Returns:
        ``np.ndarray`` формы ``(N,)`` — R² в диапазоне [0, 1]; **NaN**, где таргет
        постоянен (объяснять нечего).
    """
    n_samples = target.shape[0]
    assert predictors.shape[0] == n_samples, "N у таргета и предикторов должно совпадать"
    assert predictors.shape[-2:] == target.shape[-2:], "пространственные оси должны совпадать"

    y = target.reshape(n_samples, -1)
    design = predictors.reshape(n_samples, predictors.shape[1], -1).transpose(0, 2, 1)
    intercept = np.ones((n_samples, design.shape[1], 1), dtype=design.dtype)
    design = np.concatenate([design, intercept], axis=-1)

    gram = design.transpose(0, 2, 1) @ design
    gram += RIDGE * np.eye(gram.shape[-1], dtype=gram.dtype)[None]
    moment = design.transpose(0, 2, 1) @ y[..., None]
    beta = np.linalg.solve(gram, moment)

    residual = y - (design @ beta)[..., 0]
    ss_residual = (residual**2).sum(axis=-1)
    ss_total = ((y - y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    return np.divide(
        ss_total - ss_residual,
        ss_total,
        out=np.full(ss_total.shape, np.nan, dtype=np.float64),
        where=ss_total > 0,
    )


def regime_contrast(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Средние ``field`` внутри и вне режима ``mask`` — тест на «ремонт стока».

    Если Q_θ действительно компенсирует одностороннюю сушку ядра, его влажностный
    выход обязан быть **систематически положительнее там, где ядро сушит** (насыщенный
    подъём, δ = 1). Если контраст около нуля, Q_θ к режиму конденсации безразличен, и
    версия «ремонт» не проходит.

    Args:
        field: ``np.ndarray`` любой формы — выход Q_θ (влажностные каналы).
        mask: ``np.ndarray`` той же формы, булев — режим ядра (δ = 1).

    Returns:
        ``dict`` с ключами ``inside``, ``outside``, ``contrast`` (inside − outside) и
        ``mask_fraction``. Средние — NaN, если режим пуст или занимает всё поле.
    """
    assert field.shape == mask.shape, f"формы не совпадают: {field.shape} vs {mask.shape}"
    inside = mask.astype(bool)
    n_inside = int(inside.sum())
    n_total = int(inside.size)

    mean_inside = float(field[inside].mean()) if n_inside else np.nan
    mean_outside = float(field[~inside].mean()) if n_inside < n_total else np.nan
    return {
        "inside": mean_inside,
        "outside": mean_outside,
        "contrast": mean_inside - mean_outside,
        "mask_fraction": n_inside / n_total,
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    """Сводка распределения (NaN игнорируются): среднее, медиана, доля значимых знаков.

    Args:
        values: ``np.ndarray`` — например, корреляции по (сэмпл × шаг × уровень).

    Returns:
        ``dict`` со средним, медианой, стандартным отклонением, долями значений выше
        +0.3 и ниже −0.3 (порог «заметной» корреляции) и долей валидных элементов.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "frac_pos": np.nan,
            "frac_neg": np.nan,
            "frac_valid": 0.0,
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "frac_pos": float((finite > 0.3).mean()),
        "frac_neg": float((finite < -0.3).mean()),
        "frac_valid": float(finite.size / values.size),
    }
