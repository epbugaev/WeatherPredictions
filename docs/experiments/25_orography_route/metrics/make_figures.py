"""Хитмапы метрик exp25 (изоляция маршрута орографии) относительно baseline d0.

exp25 = 2×2 (орография в диабатике: вкл/выкл) × (орография каналом: вкл/выкл), все
армы PI-IAM4VP, USA, seed 0, эпоха 500. Хитмапы «уровень × шаг» показывают выигрыш
каждого маршрута над d0 (орографии нет нигде):

* d1 — орография только в диабатике Q_θ;
* d2 — орография только входным каналом;
* d3 — орография в обоих (избыточность).

Прямой ответ на вопрос эксперимента: сравнить d1 (диабатика) и d2 (канал) к d0.
Отрисовка общая с exp16/20/21/22/24 (:mod:`tools.metrics_ladder`).

Вход — ``results/iam4vp_usa/`` с ``metrics_<arm>.npz`` + ``paired_deltas.npz``
(скачаны с кластера, см. ``sh_files/exp25_metrics_eval.sh``; файл
``paired_deltas_iam4vp_usa.npz`` переименовать в ``paired_deltas.npz``).

Запуск (локально):
    python docs/experiments/25_orography_route/metrics/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import LadderSpec, write_metric_outputs  # noqa: E402

REGION = "usa"
EPOCH = 500
# Okabe-Ito: контроль серый, диабатика синяя, канал оранжевый, оба зелёный.
ARM_COLORS = {"d0": "#8a8a8a", "d1": "#0072B2", "d2": "#E69F00", "d3": "#009E73"}


def build_spec() -> LadderSpec:
    """Спецификация лестницы exp25: baseline d0, армы d1 (диабатика)/d2 (канал)/d3 (оба)."""
    key = f"exp25-iam4vp-{{arm}}-{REGION}"
    arms = {tag: key.format(arm=tag) for tag in ARM_COLORS}
    label_text = {
        "d0": "d0 — орографии нет",
        "d1": "d1 — оро в диабатике",
        "d2": "d2 — оро каналом",
        "d3": "d3 — оро в обоих",
    }
    labels = {arms[tag]: label_text[tag] for tag in ARM_COLORS}
    baseline = arms["d0"]
    arm_order = (arms["d1"], arms["d2"], arms["d3"])
    return LadderSpec(
        baseline=baseline,
        baseline_label="d0 (без орографии)",
        arm_order=arm_order,
        labels=labels,
        level_arms=arm_order,
        psd_arms=(baseline, arms["d1"], arms["d2"], arms["d3"]),
        psd_colors={arms[tag]: ARM_COLORS[tag] for tag in ARM_COLORS},
        epoch=EPOCH,
        table_title=f"# Метрики exp25 — маршрут орографии, PI-IAM4VP, USA (эпоха {EPOCH})",
        arm_filename_prefix="exp25-iam4vp-",
    )


def main() -> None:
    """CLI: результаты exp25 → форест/levels/psd/хитмапы в results/iam4vp_usa/."""
    results_dir = HERE / "results" / f"iam4vp_{REGION}"
    write_metric_outputs(
        spec=build_spec(),
        results_dir=results_dir,
        figures_dir=results_dir,
        heatmaps_dir=results_dir / "heatmaps",
    )


if __name__ == "__main__":
    main()
