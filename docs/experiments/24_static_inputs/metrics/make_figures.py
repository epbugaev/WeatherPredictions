"""Хитмапы метрик лестницы exp24 (IAM4VP, статические входы) относительно
контроля ``no_physics + orography``.

exp24 сравнивает вклад статических каналов и физического приора A2 поверх
орографии. Baseline лестницы — ``no_physics+orog`` (без физики, но с рельефом),
как просил пользователь: хитмапы показывают выигрыш каждого арма именно над ним.
Три арма к контролю:

* ``a2+orog``     — вклад физики A2 поверх орографии (тест H2 спека);
* ``no_physics+static`` — вклад канала суша/море (lsm) поверх орографии;
* ``a2+static``   — физика + lsm вместе.

Отрисовка общая с exp16/20/21/22 (:mod:`tools.metrics_ladder`); здесь — только
спецификация лестницы и пути. Форест (``fig_ci_forest``) НЕ строится: его подпись
утверждает «все армы на ОБЩЕЙ эпохе», а армы exp24 стоят на разных эпохах last.pt
(обучение оборвано концом proj_1715). Эпоха каждого арма вынесена в его подпись;
разброс эпох ≫ эффекта статики — оговорка в README рядом.

Вход — ``results/iam4vp_usa/`` с ``metrics_<arm>.npz`` + ``paired_deltas.npz``
(скачаны с кластера, см. ``sh_files/exp24_metrics_eval.sh``; файл
``paired_deltas_iam4vp_usa.npz`` переименовать в ``paired_deltas.npz``).

Запуск (локально):
    python docs/experiments/24_static_inputs/metrics/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import (  # noqa: E402
    METRICS,
    LadderSpec,
    load,
    render_arm_metric_heatmap,
    render_levels,
    render_psd,
    write_table,
)

REGION = "usa"
# Эпохи last.pt на момент eval (обучение оборвано, не выровнено): контроль 369,
# армы 249/429/279. Идут в подписи армов — единственный честный сигнал per-figure.
ARM_EPOCH = {"nophys-orog": 369, "a2-orog": 249, "nophys-static": 429, "a2-static": 279}
# Okabe-Ito: контроль серый, физика синяя, статика-без-физики бирюзовая.
ARM_COLORS = {
    "nophys-orog": "#8a8a8a",
    "a2-orog": "#0072B2",
    "nophys-static": "#009E73",
    "a2-static": "#D55E00",
}


def build_spec() -> LadderSpec:
    """Спецификация лестницы exp24: baseline ``no_physics+orog``, 3 арма над ним."""
    key = f"exp24-iam4vp-{{arm}}-{REGION}"
    arms = {tag: key.format(arm=tag) for tag in ARM_EPOCH}
    label_text = {
        "nophys-orog": "no_physics+orog",
        "a2-orog": "A2 exp13 + orog",
        "nophys-static": "no_physics + static (orog+lsm)",
        "a2-static": "A2 exp13 + static (orog+lsm)",
    }
    labels = {arms[tag]: f"{label_text[tag]} (эп. {ARM_EPOCH[tag]})" for tag in ARM_EPOCH}
    baseline = arms["nophys-orog"]
    arm_order = (arms["a2-orog"], arms["nophys-static"], arms["a2-static"])
    return LadderSpec(
        baseline=baseline,
        baseline_label="no_physics+orog",
        arm_order=arm_order,
        labels=labels,
        level_arms=arm_order,
        psd_arms=(baseline, arms["a2-orog"], arms["nophys-static"], arms["a2-static"]),
        psd_colors={arms[tag]: ARM_COLORS[tag] for tag in ARM_EPOCH},
        epoch=ARM_EPOCH["nophys-orog"],  # эпоха контроля; арм-эпохи — в подписях армов
        table_title=(
            "# Метрики exp24 — PI-IAM4VP, статические входы, USA "
            "(last.pt, эпохи армов РАЗНЫЕ — см. подписи)"
        ),
        arm_filename_prefix="exp24-iam4vp-",
    )


def main() -> None:
    """CLI: результаты лестницы exp24 → хитмапы уровень×шаг + таблица. Пишет PNG/markdown."""
    spec = build_spec()
    results_dir = HERE / "results" / f"iam4vp_{REGION}"
    heatmaps_dir = results_dir / "heatmaps"
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
        f"[exp24-figures] {len(summaries)} армов → сводные fig_levels_* + "
        f"пер-армовые хитмапы в {heatmaps_dir}/<метрика>/<арм>.png + таблица"
    )


if __name__ == "__main__":
    main()
