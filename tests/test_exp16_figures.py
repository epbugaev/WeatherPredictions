"""Пины чистой логики exp16 make_figures: эпох-маппинг, выбор стата, таблица.

Плоттинг (savefig) не тестируется — только преобразование метрик, где живут баги.
Модуль грузится по пути: ``docs/experiments`` не является пакетом.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "docs/experiments/16_model_ablation_ladder/make_figures.py"

_spec = importlib.util.spec_from_file_location("exp16_make_figures", MODULE_PATH)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def test_parse_var_stat() -> None:
    assert mf.parse_var_stat("RMSE_z_mean") == ("z", "mean")
    assert mf.parse_var_stat("RMSE_t_last") == ("t", "last")
    assert mf.parse_var_stat("RMSE_z_500_first") == ("z_500", "first")
    assert mf.parse_var_stat("val_loss") is None
    assert mf.parse_var_stat("RMSE_u") == ("u", "")


def test_map_points_to_epochs_uses_val_period_and_sorts() -> None:
    # намеренно не по порядку по step → должно отсортироваться
    points = [[366, 0.4], [183, 0.5], [549, 0.3]]
    mapped = mf.map_points_to_epochs(points, val_every_n_epochs=3)
    assert mapped == [(3, 0.5), (6, 0.4), (9, 0.3)]


def test_select_rmse_series_prefers_mean_over_last() -> None:
    run_metrics = {
        "RMSE_z_mean": [[183, 1.0], [366, 0.8]],
        "RMSE_z_last": [[183, 9.0], [366, 9.0]],
        "RMSE_t_last": [[183, 2.0], [366, 1.5]],
        "val_loss": [[183, 0.5]],
    }
    series = mf.select_rmse_series(run_metrics, val_every_n_epochs=3)
    # z берёт mean (не last), t — только last
    assert series["z"] == [(3, 1.0), (6, 0.8)]
    assert series["t"] == [(3, 2.0), (6, 1.5)]
    assert "val_loss" not in series


def test_base_variable_strips_level() -> None:
    assert mf.base_variable("z500") == "z"
    assert mf.base_variable("t2") == "t"
    assert mf.base_variable("u10") == "u"
    assert mf.base_variable("tp") == "tp"


def test_select_rmse_series_averages_levels() -> None:
    # два уровня одной переменной → одна кривая = среднее по уровням поэпохно
    run_metrics = {
        "RMSE_z500_mean": [[183, 1.0], [366, 0.6]],
        "RMSE_z850_mean": [[183, 3.0], [366, 1.4]],
    }
    series = mf.select_rmse_series(run_metrics, val_every_n_epochs=3)
    assert series["z"] == [(3, 2.0), (6, 1.0)]


def test_build_final_table_bolds_column_min_and_labels_arms() -> None:
    parsed = {
        "abl16-r0-no-physics-s0": {"z": [(3, 5.0), (6, 4.0)], "t": [(3, 2.0), (6, 1.8)]},
        "abl16-r5-exp15-s0": {"z": [(3, 3.0), (6, 2.0)], "t": [(3, 2.5), (6, 2.4)]},
    }
    table = mf.build_final_table(parsed, ["z", "t"])
    # финальная эпоха 6 в обеих строках
    assert "| 6 |" in table
    # min по z — 2.0 (R5), по t — 1.8 (R0): выделены жирным
    assert "**2**" in table
    assert "**1.8**" in table
    # подписи армов присутствуют
    assert "R0 · без физики" in table
    assert "R5 · +exp15" in table


def test_build_final_table_handles_missing_variable() -> None:
    parsed = {"abl16-r0-no-physics-s0": {"z": [(3, 5.0)]}}
    table = mf.build_final_table(parsed, ["z", "t"])
    # у арма нет t → прочерк, без падения
    assert "—" in table
