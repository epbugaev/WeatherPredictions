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
специфика exp21: армы, подписи, эпоха, пути. Дельты — из склейки
``merge_physics_baselines.py``: каждый арм меряется к контролю «без физики» СВОЕЙ
связки (batched → S0, S3c → S0c), поэтому форест изолирует вклад физики, как у exp16,
и не тонет в эффекте связки `chained` (см. `SPEC` ниже и §8.5 README).

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
#
# ВАЖНО — два контроля «без физики». Дельты берутся из склейки
# ``merge_physics_baselines.py``: batched-армы меряны к S0, а S3c — к S0c (контролю
# СВОЕЙ связки). Так форест показывает только вклад физики, как у exp16, и связка
# `chained` (−14.7 % сама по себе, §8.5) не топит лестницу. Поэтому S0/S0c — контроли,
# а не армы фореста: в `arm_order`/`level_arms` их нет (как нет R0 у exp16).
SPEC = LadderSpec(
    baseline="exp21L-s0-no-physics",
    baseline_label="без физики",
    arm_order=(
        "exp21L-s1-legacy-hybrid",
        "exp21L-s3a-no-diabatic",
        "exp21L-s3-a2-exp13",
        "exp21L-s4-exp14",
        "exp21L-s5-exp15",
        "exp21L-s3c-a2-exp13-chained",
    ),
    labels={
        "exp21L-s0-no-physics": "S0 · без физики",
        "exp21L-s1-legacy-hybrid": "S1 · легаси-hybrid (к S0)",
        "exp21L-s3a-no-diabatic": "S3a · A2 без Q_θ (к S0)",
        "exp21L-s3-a2-exp13": "S3 · A2 exp13 (к S0)",
        "exp21L-s4-exp14": "S4 · +exp14 (к S0)",
        "exp21L-s5-exp15": "S5 · +exp15 (к S0)",
        "exp21L-s3c-a2-exp13-chained": "S3c · A2 chained (к S0c)",
    },
    # Хитмапы [уровень × шаг] — физическая лестница: batched-армы (к S0) и
    # изолированная физика на chained (S3c к S0c). Легаси и контроли исключены,
    # как r2..r5 без R1/R0 у exp16.
    level_arms=(
        "exp21L-s3a-no-diabatic",
        "exp21L-s3-a2-exp13",
        "exp21L-s4-exp14",
        "exp21L-s5-exp15",
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
