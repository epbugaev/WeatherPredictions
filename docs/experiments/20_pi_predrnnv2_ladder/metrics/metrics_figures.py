"""Фигуры и таблицы полного набора метрик exp20 (эпоха 40, все армы на общей эпохе).

Вход: ``results/metrics_<arm>.npz`` (сводки от `metrics_eval.py`) + ``results/paired_deltas.npz``
(парные дельты к P0, ``paired_deltas.py`` эксперимента 16 с ``--baseline exp20-p0-no-physics``).
Выход:

  * ``fig_ci_forest.png`` — главная: дельта каждого арма к P0 по всем метрикам с 95% CI;
  * ``fig_levels_<метрика>.png`` — сводный хитмап [уровень × шаг], строка на арм;
  * ``../results/<метрика>/<арм>.png`` — пер-армовый хитмап [уровень × шаг] с числами;
  * ``fig_psd.png`` — спектр по зональному волновому числу m;
  * ``results/metrics_table.md`` — те же числа текстом.

Вся отрисовка общая с exp16 и живёт в :mod:`tools.metrics_ladder`; здесь — только
специфика exp20: армы, подписи, эпоха, пути.

Запуск (локально):
    .venv-exp20/bin/python docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_figures.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import LadderSpec, write_metric_outputs  # noqa: E402

# Ключи — канонические (`tools.metrics_ladder.canonical_arm`): суффикс сида срезан,
# ``exp20-p0-no-physics-s0`` → ``exp20-p0-no-physics``. Цвета — Okabe-Ito, те же, что
# на кривых val-RMSE (`../make_figures.py`), чтобы армы читались одинаково везде.
SPEC = LadderSpec(
    baseline="exp20-p0-no-physics",
    baseline_label="P0",
    arm_order=(
        "exp20-p1-legacy-hybrid",
        "exp20-p3a-no-diabatic",
        "exp20-p3-a2-exp13",
        "exp20-p4-exp14",
        "exp20-p5-exp15",
    ),
    labels={
        "exp20-p0-no-physics": "P0 · без физики",
        "exp20-p1-legacy-hybrid": "P1 · легаси-hybrid",
        "exp20-p3a-no-diabatic": "P3a · A2 без Q_θ",
        "exp20-p3-a2-exp13": "P3 · A2 (exp13)",
        "exp20-p4-exp14": "P4 · +exp14",
        "exp20-p5-exp15": "P5 · +exp15",
    },
    level_arms=(
        "exp20-p3a-no-diabatic",
        "exp20-p3-a2-exp13",
        "exp20-p4-exp14",
        "exp20-p5-exp15",
    ),
    psd_arms=("exp20-p0-no-physics", "exp20-p3-a2-exp13", "exp20-p1-legacy-hybrid"),
    psd_colors={
        "exp20-p0-no-physics": "#8a8a8a",
        "exp20-p3-a2-exp13": "#0072B2",
        "exp20-p1-legacy-hybrid": "#E69F00",
    },
    epoch=40,
    table_title="# Метрики exp20 — все армы на общей эпохе 40",
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
        help="корень для <метрика>/<арм>.png (по умолч. results/ эксперимента 20)",
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
