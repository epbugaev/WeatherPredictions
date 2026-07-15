"""Фигуры 12-шагового rollout exp20 (PI-PredRNNv2) из npz-файлов ``rollout_eval.py``.

Вход: ``results/rollout/rollout_exp20-<arm>-s0.npz`` (rmse_free/rmse_forced
(12, 69), channels, n_samples, checkpoint_epoch — у всех армов ОДНА эпоха 40,
что и делает кросс-армовое сравнение честным). Выход:
``fig_rollout_ratio_p0.png``, ``fig_rollout_delta_steps.png``,
``fig_rollout_heatmap.png`` плюс пер-армовые ``fig_rollout_heatmap_<arm>.png`` и
``fig_rollout_levels_<arm>.png``.

Нативный горизонт PredRNNv2 здесь равен полному rollout (12 шагов = один форвард),
поэтому шва скользящего окна нет: ядро (:mod:`tools.rollout_ladder`) видит
``native_horizon == steps`` и не рисует ни линию границы, ни «окно 2» в подписях.

Вся отрисовка общая с exp16 и живёт в :mod:`tools.rollout_ladder`; здесь — только
специфика exp20: армы, подписи, цвета (Okabe-Ito, те же, что на кривых val-RMSE и
в метриках) и правило канонизации имён ранов (``exp20-<arm>-s0`` → ``<arm>``).

Запуск (локально):
    .venv-exp20/bin/python docs/experiments/20_pi_predrnnv2_ladder/rollout_figures.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.rollout_ladder import RolloutSpec, write_rollout_outputs  # noqa: E402

ROLLOUT_DIR = HERE / "results" / "rollout"

SPEC = RolloutSpec(
    baseline="p0-no-physics",
    baseline_label="P0",
    arm_order=(
        "p0-no-physics",
        "p1-legacy-hybrid",
        "p3a-no-diabatic",
        "p3-a2-exp13",
        "p4-exp14",
        "p5-exp15",
    ),
    labels={
        "p0-no-physics": "P0 · без физики",
        "p1-legacy-hybrid": "P1 · легаси-hybrid",
        "p3a-no-diabatic": "P3a · A2 без Q_θ",
        "p3-a2-exp13": "P3 · A2 (exp13)",
        "p4-exp14": "P4 · +exp14",
        "p5-exp15": "P5 · +exp15",
    },
    colors={
        "p0-no-physics": "#8a8a8a",
        "p1-legacy-hybrid": "#E69F00",
        "p3a-no-diabatic": "#56B4E9",
        "p3-a2-exp13": "#0072B2",
        "p4-exp14": "#D55E00",
        "p5-exp15": "#CC79A7",
    },
    # пер-армовый хитмап — на каждый физарм лестницы (контроль не сравнивается с собой)
    heatmap_arms=(
        "p1-legacy-hybrid",
        "p3a-no-diabatic",
        "p3-a2-exp13",
        "p4-exp14",
        "p5-exp15",
    ),
    # армы исправленной физики: хитмап с числами + markdown-таблица по уровням
    level_arms=("p3a-no-diabatic", "p3-a2-exp13", "p4-exp14", "p5-exp15"),
    run_prefixes=("exp20-",),
    run_suffixes=("-s0",),
)


def main() -> None:
    """CLI: каталог rollout_*.npz → фигуры, пер-армовые хитмапы, таблица и JSON-индекс."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default=str(ROLLOUT_DIR), help="каталог rollout_*.npz")
    parser.add_argument("--suffix", default="", help="суффикс имён PNG (напр. _s1)")
    args = parser.parse_args()

    write_rollout_outputs(
        spec=SPEC,
        rollout_dir=Path(args.rollout_dir),
        figures_dir=HERE,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
