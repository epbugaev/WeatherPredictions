"""Хитмапы метрик лестницы exp24 (IAM4VP, статические входы) относительно
выбранного контроля — ``no_physics+orography`` (дефолт) или ``no_physics+static``.

exp24 сравнивает вклад статических каналов и физического приора A2. Контроль
лестницы выбирается флагом ``--baseline``:

* ``nophys-orog`` (дефолт, волна a3de087) — рельеф зафиксирован, карты читают
  добавочный вклад физики/lsm поверх орографии;
* ``nophys-static`` — контроль держит и рельеф, и маску суша/море: карты читают
  вклад ФИЗИКИ (a2-orog, a2-static) и «минус-lsm» (nophys-orog) относительно
  полного статического входа. Именно это просил пользователь.

Сид выбирается ``--seed`` (``s0`` — первичная оборванная волна, разные эпохи;
``s1`` — повторная волна на proj_1717, полная эпоха 500). Эпохи чекпоинтов
читаются из самих ``metrics_<arm>.npz`` (поле ``checkpoint_epoch``) и выносятся
в подписи армов — единственный честный per-figure сигнал при разных эпохах.

Отрисовка общая с exp16/20/21/22 (:mod:`tools.metrics_ladder`); здесь — только
спецификация лестницы и пути. Форест (``fig_ci_forest``) НЕ строится: его подпись
утверждает «все армы на ОБЩЕЙ эпохе». Для s0 это неверно, для s1 верно, но чтобы
не расходиться между сидами — вынесли эпоху в подписи и опустили форест везде.

Вход — ``results/<results_subdir>/`` с ``metrics_<arm>.npz`` + ``paired_deltas.npz``
(скачаны с кластера, см. ``sh_files/exp24_metrics_eval.sh``; парные дельты должны
быть пересчитаны с тем же ``--baseline``, что и здесь).

Запуск (локально):
    python docs/experiments/24_static_inputs/metrics/make_figures.py \
        --baseline nophys-static --seed s1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import (  # noqa: E402
    METRICS,
    LadderSpec,
    canonical_arm,
    load,
    render_arm_metric_heatmap,
    render_levels,
    render_psd,
    write_table,
)

REGION = "usa"
ALL_ARMS = ("nophys-orog", "a2-orog", "nophys-static", "a2-static")
# Okabe-Ito: контроль-орография серый, физика синяя/оранжевая, статика бирюзовая.
ARM_COLORS = {
    "nophys-orog": "#8a8a8a",
    "a2-orog": "#0072B2",
    "nophys-static": "#009E73",
    "a2-static": "#D55E00",
}
ARM_LABEL = {
    "nophys-orog": "no_physics+orog",
    "a2-orog": "A2 exp13 + orog",
    "nophys-static": "no_physics + static (orog+lsm)",
    "a2-static": "A2 exp13 + static (orog+lsm)",
}
# Порядок армов над контролем: сперва прочие статик-контроли, затем физика.
ARM_ORDER = {
    "nophys-orog": ("nophys-static", "a2-orog", "a2-static"),
    "nophys-static": ("nophys-orog", "a2-orog", "a2-static"),
}


def read_arm_epochs(results_dir: Path) -> dict[str, int]:
    """Прочитать ``checkpoint_epoch`` каждого арма из ``metrics_<arm>.npz``.

    Args:
        results_dir: каталог результатов лестницы.

    Returns:
        ``{канонический_арм: эпоха}`` для всех найденных ``metrics_*.npz``.
    """
    epochs: dict[str, int] = {}
    for path in sorted(results_dir.glob("metrics_*.npz")):
        data = np.load(path, allow_pickle=False)
        epochs[canonical_arm(str(data["arm"]))] = int(data["checkpoint_epoch"])
    return epochs


def build_spec(baseline_tag: str, results_dir: Path) -> LadderSpec:
    """Спецификация лестницы exp24 для заданного контроля.

    Args:
        baseline_tag: ключ контроля (``nophys-orog`` / ``nophys-static``).
        results_dir: каталог результатов (нужен для чтения эпох из npz).

    Returns:
        :class:`LadderSpec` с контролем, порядком армов, подписями (с эпохами),
        цветами PSD.
    """
    key = f"exp24-iam4vp-{{arm}}-{REGION}"
    canon = {tag: key.format(arm=tag) for tag in ALL_ARMS}
    epochs = read_arm_epochs(results_dir)
    labels = {
        canon[tag]: f"{ARM_LABEL[tag]} (эп. {epochs.get(canon[tag], '?')})" for tag in ALL_ARMS
    }
    arm_tags = ARM_ORDER[baseline_tag]
    arm_order = tuple(canon[tag] for tag in arm_tags)
    return LadderSpec(
        baseline=canon[baseline_tag],
        baseline_label=ARM_LABEL[baseline_tag],
        arm_order=arm_order,
        labels=labels,
        level_arms=arm_order,
        psd_arms=(canon[baseline_tag], *arm_order),
        psd_colors={canon[tag]: ARM_COLORS[tag] for tag in ALL_ARMS},
        epoch=epochs.get(canon[baseline_tag], 0),  # эпоха контроля; арм-эпохи — в подписях
        table_title=(
            f"# Метрики exp24 — PI-IAM4VP, статические входы, USA "
            f"(контроль {ARM_LABEL[baseline_tag]}; эпохи армов — см. подписи)"
        ),
        arm_filename_prefix="exp24-iam4vp-",
    )


def default_results_subdir(baseline_tag: str, seed: str) -> str:
    """Каталог результатов: историческое имя для (nophys-orog, s0), иначе с тегами."""
    if baseline_tag == "nophys-orog" and seed == "s0":
        return f"iam4vp_{REGION}"
    return f"iam4vp_{REGION}_{baseline_tag}_{seed}"


def main() -> None:
    """CLI: результаты лестницы exp24 → хитмапы уровень×шаг + таблица. Пишет PNG/markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", choices=("nophys-orog", "nophys-static"), default="nophys-orog"
    )
    parser.add_argument("--seed", default="s0")
    parser.add_argument(
        "--results-subdir",
        default=None,
        help="каталог под results/ (по умолчанию выводится из baseline+seed)",
    )
    args = parser.parse_args()

    subdir = args.results_subdir or default_results_subdir(args.baseline, args.seed)
    results_dir = HERE / "results" / subdir
    heatmaps_dir = results_dir / "heatmaps"
    spec = build_spec(args.baseline, results_dir)
    summaries, deltas, channels = load(results_dir, spec)

    for metric in METRICS:
        render_levels(deltas, metric, spec, results_dir / f"fig_levels_{metric}.png")
        metric_dir = heatmaps_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        for arm in spec.arm_order:
            slug = arm.removeprefix(spec.arm_filename_prefix)
            render_arm_metric_heatmap(deltas, arm, metric, spec, metric_dir / f"{slug}.png")
    render_psd(summaries, channels, spec, results_dir / "fig_psd.png")
    write_table(deltas, channels, spec, results_dir / "metrics_table.md")
    print(  # noqa: T201
        f"[exp24-figures] контроль={args.baseline} сид={args.seed}: {len(summaries)} армов → "
        f"fig_levels_* + хитмапы {heatmaps_dir}/<метрика>/<арм>.png + таблица"
    )


if __name__ == "__main__":
    main()
