"""Пины контракта имён армов в харнессе метрик (exp16 + exp20).

Ключ арма — единственная нить, связывающая три звена: ``metrics_eval.py`` кладёт в npz
``experiment.name``, ``paired_deltas.py`` канонизирует его в имена полей
``<арм>__<метрика>__delta``, а фигуры по этим ключам ищут дельты. Разъедься правило
канонизации с именами в ``LadderSpec`` — и фигуры падают на KeyError либо (хуже) молча
рисуют не тот арм. Здесь это правило и его согласованность с конфигами запинены.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from tools.metrics_ladder import METRICS, LadderSpec, canonical_arm

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP16_FIGURES = REPO_ROOT / "docs/experiments/16_model_ablation_ladder/metrics/metrics_figures.py"
EXP20_FIGURES = REPO_ROOT / "docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_figures.py"
EXP21_FIGURES = REPO_ROOT / "docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_figures.py"
EXP20_CONFIGS = REPO_ROOT / "configs/exp20"
EXP16_DELTAS = (
    REPO_ROOT / "docs/experiments/16_model_ablation_ladder/metrics/results/paired_deltas.npz"
)
EXP21_DELTAS = (
    REPO_ROOT / "docs/experiments/21_pi_simvpv2_ladder/metrics/results/paired_deltas.npz"
)


def _spec(path: Path) -> LadderSpec:
    """Загрузить ``SPEC`` скрипта фигур (``docs/experiments`` не является пакетом)."""
    spec = importlib.util.spec_from_file_location(f"figures_{path.parents[1].name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPEC


# --- правило канонизации --------------------------------------------------------


def test_canonical_arm_strips_exp16_and_exp20_decorations() -> None:
    assert canonical_arm("abl16L-r3-a2-exp13-t12-s0") == "r3-a2-exp13"
    assert canonical_arm("abl16-r0-no-physics-s0") == "r0-no-physics"
    assert canonical_arm("exp20-p3-a2-exp13-s0") == "exp20-p3-a2-exp13"
    # Префикс exp20 несёт эксперимент, а не арм: срезать его нельзя, иначе ключи
    # двух лестниц столкнутся (у обеих есть арм «p3/r3 · A2 exp13»).
    assert canonical_arm("exp20-p0-no-physics-s0").startswith("exp20-")


def test_canonical_arm_strips_exp21_seed_suffix() -> None:
    # exp21-раны названы через ``-seed0`` (long-волна ещё и ``-t12``), а не ``-s0``.
    assert canonical_arm("exp21L-s3c-a2-exp13-chained-t12-seed0") == "exp21L-s3c-a2-exp13-chained"
    assert canonical_arm("exp21-s0-no-physics-seed0") == "exp21-s0-no-physics"
    assert canonical_arm("exp21L-s0-no-physics-t12-seed0").startswith("exp21L-")


def test_canonical_arm_is_idempotent() -> None:
    for name in ("abl16L-r5-exp15-t12-s0", "exp20-p5-exp15-s0"):
        once = canonical_arm(name)
        assert canonical_arm(once) == once


# --- согласованность LadderSpec с конфигами и с уже посчитанными дельтами --------


def test_exp20_spec_keys_match_the_run_names_in_configs() -> None:
    """Каждый арм лестницы exp20 из configs/ присутствует в SPEC ровно под своим ключом."""
    spec = _spec(EXP20_FIGURES)
    run_names = [
        yaml.safe_load(path.read_text())["experiment"]["name"]
        for path in sorted(EXP20_CONFIGS.glob("exp20_*.yaml"))
    ]
    assert len(run_names) == 6
    keys = {canonical_arm(name) for name in run_names}
    assert keys == {spec.baseline, *spec.arm_order}
    assert set(spec.labels) == keys  # подпись есть у каждого арма, включая контроль
    assert set(spec.level_arms) <= set(spec.arm_order)
    assert set(spec.psd_arms) <= keys and set(spec.psd_colors) == set(spec.psd_arms)


def test_exp16_spec_keys_match_the_committed_paired_deltas() -> None:
    """Регрессия на exp16: ключи SPEC обязаны находиться в уже посчитанном npz дельт."""
    spec = _spec(EXP16_FIGURES)
    deltas = np.load(EXP16_DELTAS, allow_pickle=False)
    for arm in spec.arm_order:
        for metric in METRICS:
            assert f"{arm}__{metric}__delta" in deltas.files
            assert f"{arm}__{metric}__ci_low" in deltas.files
            assert f"{arm}__{metric}__ci_high" in deltas.files
    assert str(deltas["baseline"]) == spec.baseline


def test_exp21_spec_keys_match_the_merged_paired_deltas() -> None:
    """Регрессия на exp21: склейка двух контролей должна содержать каждый арм SPEC.

    Форест exp21 мерит физику к контролю СВОЕЙ связки (batched→S0, S3c→S0c), поэтому
    `paired_deltas.npz` — результат `merge_physics_baselines.py`. Контроли S0/S0c в
    `arm_order` не входят (как R0 у exp16), а S3c обязан присутствовать (взят из vs-S0c).
    """
    spec = _spec(EXP21_FIGURES)
    deltas = np.load(EXP21_DELTAS, allow_pickle=False)
    for arm in spec.arm_order:
        for metric in METRICS:
            assert f"{arm}__{metric}__delta" in deltas.files
            assert f"{arm}__{metric}__ci_low" in deltas.files
            assert f"{arm}__{metric}__ci_high" in deltas.files
    assert "exp21L-s3c-a2-exp13-chained__rmse__delta" in deltas.files
    assert not any(field.startswith("exp21L-s0c") for field in deltas.files)
    assert set(spec.level_arms) <= set(spec.arm_order)
    assert set(spec.labels) >= {spec.baseline, *spec.arm_order}
    assert set(spec.psd_colors) == set(spec.psd_arms)
