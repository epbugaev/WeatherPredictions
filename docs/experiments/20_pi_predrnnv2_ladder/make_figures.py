"""Фигуры и таблицы эксперимента 20 (лестница на PredRNNv2) из exp20_metrics.json.

Вход: ``results/exp20_metrics.json`` (создаётся ``collect_metrics.py``; период
валидации берётся из его ``meta.val_every_n_epochs`` — у exp20 это 2, у exp16 3).

Выход:

* ``fig_val_rmse_curves.png`` — per-variable кривые val-RMSE по эпохам, линия на арм;
* ``results/exp20_final_table.md`` — RMSE финальной эпохи;
* ``results/exp20_delta_table.md`` — Δ% к P0 «без физики» на общей эпохе;
* ``results/exp20_window_delta_table.md`` — Δ% к P0, среднее по последним 4 val-эпохам.

Обе дельта-таблицы обязательны: одноэпоховая дельта по z шумит на ±5–8 п.п.
(exp16 §8.5-а), оконная — робастный взгляд.

Вся логика общая с exp16 и живёт в :mod:`tools.ladder_figures`; здесь — только
специфика exp20: имена арм, подписи, цвета и пути.

Запуск::

    .venv-exp20/bin/python docs/experiments/20_pi_predrnnv2_ladder/make_figures.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.ladder_figures import write_ladder_outputs  # noqa: E402

# Порядок ступеней лестницы (§4 README) + подписи и цвета (Okabe-Ito, как в exp16:
# baseline — серый, легаси-hybrid — оранжевый, чтобы фигуры двух лестниц читались одинаково).
ARM_ORDER = (
    "exp20-p0-no-physics-s0",
    "exp20-p1-legacy-hybrid-s0",
    "exp20-p3a-no-diabatic-s0",
    "exp20-p3-a2-exp13-s0",
    "exp20-p4-exp14-s0",
    "exp20-p5-exp15-s0",
)
ARM_LABELS = {
    "exp20-p0-no-physics-s0": "P0 · без физики",
    "exp20-p1-legacy-hybrid-s0": "P1 · легаси-hybrid",
    "exp20-p3a-no-diabatic-s0": "P3a · A2 без Q_θ",
    "exp20-p3-a2-exp13-s0": "P3 · A2 (exp13)",
    "exp20-p4-exp14-s0": "P4 · +exp14",
    "exp20-p5-exp15-s0": "P5 · +exp15",
}
ARM_COLORS = {
    "exp20-p0-no-physics-s0": "#8a8a8a",
    "exp20-p1-legacy-hybrid-s0": "#E69F00",
    "exp20-p3a-no-diabatic-s0": "#56B4E9",
    "exp20-p3-a2-exp13-s0": "#0072B2",
    "exp20-p4-exp14-s0": "#D55E00",
    "exp20-p5-exp15-s0": "#CC79A7",
}
TITLE = "Лестница exp 20 (PredRNNv2): val-RMSE по эпохам (сид 0, вал-2004)"


def main() -> None:
    """Собрать фигуру и три таблицы из results/exp20_metrics.json."""
    write_ladder_outputs(
        metrics_path=HERE / "results" / "exp20_metrics.json",
        figure_path=HERE / "fig_val_rmse_curves.png",
        tables_dir=HERE / "results",
        table_prefix="exp20",
        arm_order=ARM_ORDER,
        labels=ARM_LABELS,
        colors=ARM_COLORS,
        title=TITLE,
    )


if __name__ == "__main__":
    main()
