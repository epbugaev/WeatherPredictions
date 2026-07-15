"""Фигуры 12-шагового rollout exp21 (PI-SimVPv2) из npz-файлов ``rollout_eval.py``.

Вход: ``results/rollout/rollout_exp21L-<arm>-t12-seed0.npz`` (rmse_free/rmse_forced
(12, 69), channels, n_samples, checkpoint_epoch — у всех армов ОДНА эпоха 500, что и
делает кросс-армовое сравнение честным). Выход: ``fig_rollout_ratio_s0.png``,
``fig_rollout_delta_steps.png``, ``fig_rollout_heatmap.png`` плюс пер-армовые
``fig_rollout_heatmap_<arm>.png`` и ``fig_rollout_levels_<arm>.png``.

Специфика exp21 против exp16/exp20:

  * **baseline единый — S0** (batched без физики), все 8 армов на ratio-панели.
    Ось отношения (~0.85–1.05) вмещает и связку, и физику: batched-лестница жмётся
    к 1.0, пара chained (S0c/S3c) уходит к ~0.85, а разрыв S3c−S0c = вклад физики
    поверх связки — видно per-лид-тайм. Физику отдельно изолирует форест метрик
    (`metrics/`, контроль своей связки); rollout же показывает лид-таймовую
    структуру ВСЕХ эффектов на одной оси.
  * **single_mode=True** — SimVPv2 MIMO: весь горизонт одним форвардом, teacher-
    forcing не определён (``rmse_forced == rmse_free``), поэтому delta_steps —
    одна панель, а не вырожденная пара.

Нативный горизонт SimVPv2 равен полному rollout (12 шагов = один форвард), поэтому
шва скользящего окна нет (``native_horizon == steps`` → ядро не рисует границу).
Вся отрисовка общая с exp16/exp20 и живёт в :mod:`tools.rollout_ladder`; здесь —
только армы, подписи, цвета (Okabe-Ito, как в `make_figures.py`) и канонизация имён.

Запуск (локально):
    python docs/experiments/21_pi_simvpv2_ladder/rollout_figures.py
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
    baseline="exp21L-s0-no-physics",
    baseline_label="S0",
    arm_order=(
        "exp21L-s0-no-physics",
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
    colors={
        "exp21L-s0-no-physics": "#8a8a8a",
        "exp21L-s1-legacy-hybrid": "#E69F00",
        "exp21L-s3a-no-diabatic": "#56B4E9",
        "exp21L-s3-a2-exp13": "#0072B2",
        "exp21L-s4-exp14": "#D55E00",
        "exp21L-s5-exp15": "#CC79A7",
        "exp21L-s0c-no-physics-chained": "#F0E442",
        "exp21L-s3c-a2-exp13-chained": "#009E73",
    },
    # Пер-армовые хитмапы — легаси, batched-лестница и оба chained-арма: контраст
    # связки (S0c) и связки-с-физикой (S3c) виден по [уровень × шаг].
    heatmap_arms=(
        "exp21L-s1-legacy-hybrid",
        "exp21L-s3-a2-exp13",
        "exp21L-s0c-no-physics-chained",
        "exp21L-s3c-a2-exp13-chained",
    ),
    level_arms=(
        "exp21L-s3-a2-exp13",
        "exp21L-s0c-no-physics-chained",
        "exp21L-s3c-a2-exp13-chained",
    ),
    run_prefixes=(),  # префикс exp21L несёт эксперимент — не срезаем (иначе ключи столкнутся)
    run_suffixes=("-t12-seed0", "-seed0"),
    single_mode=True,
)


def main() -> None:
    """CLI: каталог rollout-npz → фигуры + пер-армовые хитмапы + таблица + индекс."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=HERE)
    args = parser.parse_args()
    write_rollout_outputs(SPEC, args.rollout_dir, args.figures_dir, suffix="_t12")


if __name__ == "__main__":
    main()
