"""Фигуры и таблицы полного набора метрик exp16 (эпоха 500, все армы на общей эпохе).

Вход: ``results/metrics_<arm>.npz`` (сводки) + ``results/paired_deltas.npz`` (парные
дельты к R0 с бутстрап-CI). Выход:

  * ``fig_ci_forest.png`` — главная: дельта каждого арма к R0 по всем метрикам с 95% CI
    (значимо ⇔ интервал не накрывает ноль);
  * ``fig_levels_<метрика>.png`` — хитмап [уровень × шаг] к R0, строка на арм, столбец
    на переменную; **незначимые ячейки заштрихованы** — их читать нельзя;
  * ``../results/<метрика>/<арм>.png`` — пер-армовый хитмап [уровень × шаг] с числами;
  * ``fig_psd.png`` — спектр по зональному волновому числу m: прогноз против истины
    (падение мощности на больших m = модель сглаживает мелкий масштаб);
  * ``results/metrics_table.md`` — те же числа текстом.

Вся отрисовка общая с exp20 и живёт в :mod:`tools.metrics_ladder`; здесь — только
специфика exp16: армы, подписи, эпоха, пути.

Запуск (локально):
    python metrics_figures.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import LadderSpec, write_metric_outputs  # noqa: E402

SPEC = LadderSpec(
    baseline="r0-no-physics",
    baseline_label="R0",
    arm_order=(
        "r1-legacy-hybrid",
        "r2a-a1-pre13",
        "r2-a2-pre13",
        "r3-a2-exp13",
        "r4-exp14",
        "r5-exp15",
        "r3a-no-diabatic",
        "r3q-diabatic-t-only",
    ),
    labels={
        "r0-no-physics": "R0 · без физики",
        "r1-legacy-hybrid": "R1 · легаси",
        "r2a-a1-pre13": "R2a · A1 (без Q_θ)",
        "r2-a2-pre13": "R2 · A2 (до exp13)",
        "r3-a2-exp13": "R3 · A2 + exp13",
        "r4-exp14": "R4 · + exp14",
        "r5-exp15": "R5 · + exp15",
        "r3a-no-diabatic": "R3a · R3 без Q_θ",
        "r3q-diabatic-t-only": "R3q · Q_θ только t",
    },
    level_arms=("r2-a2-pre13", "r3-a2-exp13", "r4-exp14", "r5-exp15"),
    psd_arms=("r0-no-physics", "r3-a2-exp13", "r1-legacy-hybrid"),
    psd_colors={
        "r0-no-physics": "#8a8a8a",
        "r3-a2-exp13": "#0072B2",
        "r1-legacy-hybrid": "#E69F00",
    },
    epoch=500,
    table_title="# Метрики exp16 — все армы на общей эпохе 500",
)


def main() -> None:
    """CLI: каталог результатов → сводные фигуры, пер-армовые хитмапы, markdown-таблица.

    Пер-армовые хитмапы раскладываются как ``<heatmaps-dir>/<метрика>/<арм>.png`` —
    каталог на метрику, файл на арм.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument(
        "--heatmaps-dir",
        default=str(HERE.parent / "results"),
        help="корень для <метрика>/<арм>.png (по умолч. results/ эксперимента 16)",
    )
    args = parser.parse_args()

    write_metric_outputs(
        spec=SPEC,
        results_dir=Path(args.results_dir),
        figures_dir=HERE,
        heatmaps_dir=Path(args.heatmaps_dir),
    )


if __name__ == "__main__":
    main()
