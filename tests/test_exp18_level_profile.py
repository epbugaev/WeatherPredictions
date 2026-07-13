"""Пины чистой логики exp18: подпись эпох чекпоинтов в послойной сводке.

Инференс и рендер не тестируются — только то, что попадает в текст выводов и
может соврать. Модуль грузится по пути: ``docs/experiments`` не является пакетом.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "docs/experiments/18_level_resolved_physics"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lp_mod = _load("exp18_level_profile", "level_profile_analysis.py")


def test_epoch_label_fixed_epoch_wave() -> None:
    # волна с общего тега: все армы с одной эпохи → одна подпись (тег 89 = эпоха 90)
    runs = {"r0-no-physics": {"epoch": 89}, "r4-exp14": {"epoch": 89}}
    assert lp_mod.epoch_label(runs) == "эпоха 90"


def test_epoch_label_best_val_selection_reports_range() -> None:
    # отбор по валидации: эпохи армов разные → подпись обязана показать диапазон.
    # Иначе сводка выдаёт эпоху произвольного арма (dict-порядок!) за общую.
    runs = {"r0-no-physics": {"epoch": 269}, "r5-exp15": {"epoch": 169}}
    assert lp_mod.epoch_label(runs) == "лучший val, эпохи 170–270"


def test_epoch_label_without_provenance() -> None:
    runs = {"r0-no-physics": {"epoch": -1}}
    assert lp_mod.epoch_label(runs) == "эпоха ?"
