"""Фигуры 12-шагового rollout exp16 из npz-файлов ``rollout_eval.py``.

Вход: ``results/rollout*/rollout_<arm>.npz`` (rmse_free/rmse_forced (12, 69),
channels, n_samples). Выход — фигуры в стиле инференс-диагностики предыдущего
поколения абляции: ``fig_rollout_ratio_r0.png``, ``fig_rollout_delta_steps.png``,
``fig_rollout_heatmap.png`` плюс пер-армовые ``fig_rollout_heatmap_<arm>.png`` и
``fig_rollout_levels_<arm>.png``.

Вся отрисовка общая с экспериментом 20 и живёт в :mod:`tools.rollout_ladder`;
здесь — только специфика exp16: имена армов, подписи, цвета (Okabe-Ito), правило
канонизации имён ранов (волны t=6 ``abl16-`` и t=12 ``abl16L-``) и пути.

Функции ниже — тонкие обёртки, фиксирующие арм-константы exp16 в общих функциях
ядра. Именно их (с этими сигнатурами) пинит ``tests/test_exp16_rollout.py``.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.rollout_ladder import (  # noqa: E402
    RolloutSpec,
    delta_percent,
    epoch_label,
    level_matrix,
    mean_delta_over_channels,
    write_rollout_outputs,
)
from tools.rollout_ladder import level_step_delta as _level_step_delta  # noqa: E402
from tools.rollout_ladder import window_boundary as _window_boundary  # noqa: E402

__all__ = [
    "canonical_arm",
    "delta_percent",
    "epoch_label",
    "level_matrix",
    "level_step_delta",
    "main",
    "mean_delta_over_channels",
    "window_boundary",
]

ROLLOUT_DIR = HERE / "results" / "rollout"

# Реестр канонический: ключ независим от волны (t=6 ``abl16-<arm>-s0`` и
# t=12 ``abl16L-<arm>-t12-s0`` сводятся к одному стему ``canonical_arm``).
SPEC = RolloutSpec(
    baseline="r0-no-physics",
    baseline_label="R0",
    arm_order=(
        "r0-no-physics",
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
        "r1-legacy-hybrid": "R1 · легаси-hybrid",
        "r2a-a1-pre13": "R2a · A1 (до exp13)",
        "r2-a2-pre13": "R2 · A2 (до exp13)",
        "r3-a2-exp13": "R3 · A2 (exp13)",
        "r4-exp14": "R4 · +exp14",
        "r5-exp15": "R5 · +exp15",
        "r3a-no-diabatic": "R3a · R3 без Q_θ",
        "r3q-diabatic-t-only": "R3q · Q_θ только t",
    },
    colors={
        "r0-no-physics": "#8a8a8a",
        "r1-legacy-hybrid": "#E69F00",
        "r2a-a1-pre13": "#56B4E9",
        "r2-a2-pre13": "#009E73",
        "r3-a2-exp13": "#0072B2",
        "r4-exp14": "#D55E00",
        "r5-exp15": "#CC79A7",
        "r3a-no-diabatic": "#000000",
        "r3q-diabatic-t-only": "#AA4499",
    },
    # армы лестницы в общем хитмапе (без R0): отдельная фигура на каждый
    heatmap_arms=(
        "r1-legacy-hybrid",
        "r2a-a1-pre13",
        "r2-a2-pre13",
        "r3-a2-exp13",
        "r4-exp14",
        "r5-exp15",
    ),
    # армы с исправленными уравнениями: markdown-таблица по уровням + аннотированный хитмап
    level_arms=("r2-a2-pre13", "r3-a2-exp13", "r4-exp14", "r5-exp15"),
    run_prefixes=("abl16L-", "abl16-"),
    run_suffixes=("-t12-s0", "-s0"),
)


def canonical_arm(name: str) -> str:
    """Свести имя рана exp16 к волно-независимому стему (``abl16L-r0-…-t12-s0`` → ``r0-no-physics``).

    Args:
        name: имя эксперимента из npz (t=6 ``abl16-<arm>-s0`` или t=12
            ``abl16L-<arm>-t12-s0``).

    Returns:
        Канонический ключ арма (совпадает с ключами ``SPEC.arm_order``).
    """
    return SPEC.canonical(name)


def window_boundary(runs: dict[str, dict]) -> int | None:
    """Шаг границы окон exp16 (или ``None``, если прогноз одним нативным окном)."""
    return _window_boundary(runs, SPEC.baseline)


def level_step_delta(runs: dict[str, dict], arm: str, var: str) -> tuple[np.ndarray, list[int]]:
    """Δ% RMSE арма к R0 (free-running) на сетке [уровень давления × шаг]."""
    return _level_step_delta(runs, arm, var, SPEC.baseline)


def main() -> None:
    """Собрать rollout-фигуры, пер-армовые хитмапы, таблицу по уровням и JSON-индекс.

    CLI: ``--rollout-dir`` (по умолч. results/rollout — волна t=6 USA) и
    ``--suffix`` (добавляется к именам PNG, напр. ``_t12``), чтобы разные
    волны не затирали фигуры друг друга.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default=str(ROLLOUT_DIR), help="каталог rollout_*.npz")
    parser.add_argument("--suffix", default="", help="суффикс имён PNG (напр. _t12)")
    args = parser.parse_args()

    write_rollout_outputs(
        spec=SPEC,
        rollout_dir=Path(args.rollout_dir),
        figures_dir=HERE,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
