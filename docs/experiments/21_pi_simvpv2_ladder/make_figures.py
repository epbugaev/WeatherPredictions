"""Фигуры и таблицы эксперимента 21 (лестница на SimVPv2) из exp21_metrics.json.

Вход: ``results/exp21_metrics.json`` (создаётся ``collect_metrics.py``; период
валидации берётся из его ``meta.val_every_n_epochs`` — у exp21-long это 10).

Выход:

* ``fig_val_rmse_curves.png`` — per-variable кривые val-RMSE по эпохам, линия на арм;
* ``fig_delta_bars.png`` — столбцы Δ% к S0 (снимок общей эпохи + окно эпох);
* ``results/exp21_final_table.md`` — RMSE финальной эпохи;
* ``results/exp21_delta_table.md`` — Δ% к S0 «без физики» на общей эпохе;
* ``results/exp21_window_delta_table.md`` — Δ% к S0, среднее по последним 4 val-эпохам.

Обе дельта-таблицы обязательны: одноэпоховая дельта по z шумит на ±5–8 п.п.
(exp16 §8.5-а), оконная — робастный взгляд.

Вся логика общая с exp16/exp20 и живёт в :mod:`tools.ladder_figures`; здесь —
только специфика exp21: имена арм, подписи, цвета и пути.

Запуск::

    python docs/experiments/21_pi_simvpv2_ladder/make_figures.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.ladder_figures import write_ladder_outputs  # noqa: E402

# Порядок ступеней лестницы (§4 README): baseline первым (он же контроль дельт),
# затем легаси и A-семейство (batched), затем контроль связки chained.
# Цвета Okabe-Ito, как в exp16/exp20: baseline — серый, легаси — оранжевый.
# Пара chained (S0c/S3c) — главная находка exp21, поэтому яркие и рядом.
ARM_ORDER = (
    "exp21L-s0-no-physics-t12-seed0",
    "exp21L-s1-legacy-hybrid-t12-seed0",
    "exp21L-s3a-no-diabatic-t12-seed0",
    "exp21L-s3-a2-exp13-t12-seed0",
    "exp21L-s4-exp14-t12-seed0",
    "exp21L-s5-exp15-t12-seed0",
    "exp21L-s0c-no-physics-chained-t12-seed0",
    "exp21L-s3c-a2-exp13-chained-t12-seed0",
)
ARM_LABELS = {
    "exp21L-s0-no-physics-t12-seed0": "S0 · без физики",
    "exp21L-s1-legacy-hybrid-t12-seed0": "S1 · легаси-hybrid",
    "exp21L-s3a-no-diabatic-t12-seed0": "S3a · A2 без Q_θ",
    "exp21L-s3-a2-exp13-t12-seed0": "S3 · A2 (exp13)",
    "exp21L-s4-exp14-t12-seed0": "S4 · +exp14",
    "exp21L-s5-exp15-t12-seed0": "S5 · +exp15",
    "exp21L-s0c-no-physics-chained-t12-seed0": "S0c · chained без физики",
    "exp21L-s3c-a2-exp13-chained-t12-seed0": "S3c · A2 chained",
}
ARM_COLORS = {
    "exp21L-s0-no-physics-t12-seed0": "#8a8a8a",
    "exp21L-s1-legacy-hybrid-t12-seed0": "#E69F00",
    "exp21L-s3a-no-diabatic-t12-seed0": "#56B4E9",
    "exp21L-s3-a2-exp13-t12-seed0": "#0072B2",
    "exp21L-s4-exp14-t12-seed0": "#D55E00",
    "exp21L-s5-exp15-t12-seed0": "#CC79A7",
    "exp21L-s0c-no-physics-chained-t12-seed0": "#F0E442",
    "exp21L-s3c-a2-exp13-chained-t12-seed0": "#009E73",
}
TITLE = "Лестница exp 21 (SimVPv2): val-RMSE по эпохам (сид 0, вал-2004)"


def main() -> None:
    """Собрать фигуры (кривые + столбцы дельт) и три таблицы из exp21_metrics.json."""
    write_ladder_outputs(
        metrics_path=HERE / "results" / "exp21_metrics.json",
        figure_path=HERE / "fig_val_rmse_curves.png",
        tables_dir=HERE / "results",
        table_prefix="exp21",
        arm_order=ARM_ORDER,
        labels=ARM_LABELS,
        colors=ARM_COLORS,
        title=TITLE,
        bars_path=HERE / "fig_delta_bars.png",
    )


if __name__ == "__main__":
    main()
