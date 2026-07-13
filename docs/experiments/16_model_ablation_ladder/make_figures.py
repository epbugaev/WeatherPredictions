"""Фигуры и таблицы эксперимента 16 (модельная лестница) из abl16_metrics.json.

Вход: ``results/abl16_metrics.json`` (создаётся ``collect_metrics.py``).
Выход: ``fig_val_rmse_curves.png`` (per-variable кривые val-RMSE по эпохам, линия
на арм) и таблицы ``results/abl16_{final,delta,window_delta}_table.md``.

Вся логика (эпох-маппинг, выбор стата, построение таблиц, отрисовка) общая с
экспериментом 20 и живёт в :mod:`tools.ladder_figures`; здесь — только специфика
exp16: имена арм, подписи, цвета (Okabe-Ito), пути и заголовок фигуры.

Функции ниже — тонкие обёртки, фиксирующие арм-константы exp16 в общих функциях
ядра. Именно их (с этими сигнатурами) пинит ``tests/test_exp16_figures.py``.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.ladder_figures import (  # noqa: E402
    base_variable,
    common_epoch,
    map_points_to_epochs,
    order_variables,
    parse_var_stat,
    select_rmse_series,
    write_ladder_outputs,
)
from tools.ladder_figures import build_delta_table as _build_delta_table  # noqa: E402
from tools.ladder_figures import build_final_table as _build_final_table  # noqa: E402
from tools.ladder_figures import build_window_delta_table as _build_window_delta_table  # noqa: E402

__all__ = [
    "base_variable",
    "build_delta_table",
    "build_final_table",
    "build_window_delta_table",
    "common_epoch",
    "main",
    "map_points_to_epochs",
    "order_variables",
    "parse_var_stat",
    "select_rmse_series",
]

# Порядок ступеней лестницы + человекочитаемые подписи и цвета (Okabe-Ito).
ARM_ORDER = (
    "abl16-r0-no-physics-s0",
    "abl16-r1-legacy-hybrid-s0",
    "abl16-r2a-a1-pre13-s0",
    "abl16-r2-a2-pre13-s0",
    "abl16-r3-a2-exp13-s0",
    "abl16-r4-exp14-s0",
    "abl16-r5-exp15-s0",
)
ARM_LABELS = {
    "abl16-r0-no-physics-s0": "R0 · без физики",
    "abl16-r1-legacy-hybrid-s0": "R1 · легаси-hybrid",
    "abl16-r2a-a1-pre13-s0": "R2a · A1 (до exp13)",
    "abl16-r2-a2-pre13-s0": "R2 · A2 (до exp13)",
    "abl16-r3-a2-exp13-s0": "R3 · A2 (exp13)",
    "abl16-r4-exp14-s0": "R4 · +exp14",
    "abl16-r5-exp15-s0": "R5 · +exp15",
}
ARM_COLORS = {
    "abl16-r0-no-physics-s0": "#8a8a8a",
    "abl16-r1-legacy-hybrid-s0": "#E69F00",
    "abl16-r2a-a1-pre13-s0": "#56B4E9",
    "abl16-r2-a2-pre13-s0": "#009E73",
    "abl16-r3-a2-exp13-s0": "#0072B2",
    "abl16-r4-exp14-s0": "#D55E00",
    "abl16-r5-exp15-s0": "#CC79A7",
}
TITLE = "Лестница exp 16: val-RMSE по эпохам (сид 0, вал-2004)"


def build_final_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    variables: list[str],
) -> str:
    """RMSE финальной эпохи (арм × переменная) с армами exp16."""
    return _build_final_table(parsed_by_run, variables, ARM_ORDER, ARM_LABELS)


def build_delta_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    baseline_run: str,
    variables: list[str],
    epoch: int,
) -> str:
    """Δ% RMSE к baseline на общей эпохе, с армами exp16."""
    return _build_delta_table(parsed_by_run, baseline_run, variables, epoch, ARM_ORDER, ARM_LABELS)


def build_window_delta_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    baseline_run: str,
    variables: list[str],
    epochs: tuple[int, ...],
) -> str:
    """Δ% RMSE к baseline, среднее по окну эпох, с армами exp16."""
    return _build_window_delta_table(
        parsed_by_run, baseline_run, variables, epochs, ARM_ORDER, ARM_LABELS
    )


def main() -> None:
    """Собрать фигуру и таблицы из results/abl16_metrics.json."""
    write_ladder_outputs(
        metrics_path=HERE / "results" / "abl16_metrics.json",
        figure_path=HERE / "fig_val_rmse_curves.png",
        tables_dir=HERE / "results",
        table_prefix="abl16",
        arm_order=ARM_ORDER,
        labels=ARM_LABELS,
        colors=ARM_COLORS,
        title=TITLE,
    )


if __name__ == "__main__":
    main()
