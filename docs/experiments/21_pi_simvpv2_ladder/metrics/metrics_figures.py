"""Фигуры и таблицы полного набора метрик exp21 (эпоха 500, все армы на общей эпохе).

Вход: ``results/metrics_<arm>.npz`` (сводки от `metrics_eval.py`) + ``results/paired_deltas.npz``
(парные дельты к S0, ``paired_deltas.py`` эксперимента 16 с ``--baseline exp21L-s0-no-physics``).
Выход:

  * ``fig_ci_forest.png`` — главная: дельта каждого арма к S0 по всем метрикам с 95% CI;
  * ``fig_levels_<метрика>.png`` — сводный хитмап [уровень × шаг], строка на арм;
  * ``../results/<метрика>/<арм>.png`` — пер-армовый хитмап [уровень × шаг] с числами;
  * ``fig_psd.png`` — спектр по зональному волновому числу m;
  * ``results/metrics_table.md`` — те же числа текстом.

Вся отрисовка общая с exp16/exp20 и живёт в :mod:`tools.metrics_ladder`; здесь — только
специфика exp21: армы, подписи, эпоха, пути. Порядок и цвета — как на кривых val-RMSE
(`../make_figures.py`): baseline серый, легаси оранжевый, пара chained (S0c/S3c) яркая.

Запуск (локально):
    python docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_figures.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import LadderSpec, write_metric_outputs  # noqa: E402

# Ключи — канонические (`tools.metrics_ladder.canonical_arm`): суффикс сида срезан,
# ``exp21L-s0-no-physics-t12-seed0`` → ``exp21L-s0-no-physics``.
SPEC = LadderSpec(
    baseline="exp21L-s0-no-physics",
    baseline_label="S0",
    arm_order=(
        "exp21L-s1-legacy-hybrid",
        "exp21L-s3a-no-diabatic",
        "exp21L-s3-a2-exp13",
        "exp21L-s4-exp14",
        "exp21L-s5-exp15",
        "exp21L-s0c-no-physics-chained",
        "exp21L-s3c-a2-exp13-chained",
    ),
    labels={
        "exp21L-s0-no-physics": "S0 · без физики",
        "exp21L-s1-legacy-hybrid": "S1 · легаси-hybrid",
        "exp21L-s3a-no-diabatic": "S3a · A2 без Q_θ",
        "exp21L-s3-a2-exp13": "S3 · A2 (exp13)",
        "exp21L-s4-exp14": "S4 · +exp14",
        "exp21L-s5-exp15": "S5 · +exp15",
        "exp21L-s0c-no-physics-chained": "S0c · chained без физики",
        "exp21L-s3c-a2-exp13-chained": "S3c · A2 chained",
    },
    # Хитмапы [уровень × шаг] — по всем физ-армам и обоим chained-контролям,
    # чтобы был виден и разброс batched-лестницы, и глубокий выигрыш связки.
    level_arms=(
        "exp21L-s3a-no-diabatic",
        "exp21L-s3-a2-exp13",
        "exp21L-s4-exp14",
        "exp21L-s5-exp15",
        "exp21L-s0c-no-physics-chained",
        "exp21L-s3c-a2-exp13-chained",
    ),
    # PSD: baseline, лучший batched (S3), лучший связочный (S3c), легаси — контраст.
    psd_arms=(
        "exp21L-s0-no-physics",
        "exp21L-s3c-a2-exp13-chained",
        "exp21L-s3-a2-exp13",
        "exp21L-s1-legacy-hybrid",
    ),
    psd_colors={
        "exp21L-s0-no-physics": "#8a8a8a",
        "exp21L-s3c-a2-exp13-chained": "#009E73",
        "exp21L-s3-a2-exp13": "#0072B2",
        "exp21L-s1-legacy-hybrid": "#E69F00",
    },
    epoch=500,
    table_title="# Метрики exp21 — все армы на общей эпохе 500",
)


def main() -> None:
    """CLI: каталог результатов → сводные фигуры, пер-армовые хитмапы, markdown-таблица."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument(
        "--heatmaps-dir",
        default=str(HERE.parent / "results"),
        help="корень для <метрика>/<арм>.png (по умолч. results/ эксперимента 21)",
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
