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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import (  # noqa: E402
    METRICS,
    LadderSpec,
    canonical_arm,
    render_arm_metric_heatmap,
    render_forest,
    render_levels,
    render_psd,
    write_table,
)

REGION = "usa"
EPOCH = 500
# Okabe-Ito: контроль серый, диабатика синяя, канал оранжевый, оба зелёный.
ARM_COLORS = {"d0": "#8a8a8a", "d1": "#0072B2", "d2": "#E69F00", "d3": "#009E73"}
ARM_LABEL = {
    "d0": "d0 — орографии нет",
    "d1": "d1 — оро в диабатике",
    "d2": "d2 — оро каналом",
    "d3": "d3 — оро в обоих",
}


def present_arms(results_dir: Path) -> list[str]:
    """Теги армов (кроме baseline d0), у которых есть парные дельты в npz.

    Промежуточный прогон считает d1/d2 без d3 — форест/хитмапы должны включать
    только реально посчитанные армы, иначе delta_grid падает по отсутствующему ключу.
    """
    deltas = np.load(results_dir / "paired_deltas.npz", allow_pickle=False)
    keys = set(deltas.files)
    return [t for t in ("d1", "d2", "d3") if f"exp25-iam4vp-{t}-{REGION}__rmse__delta" in keys]


def build_spec(tags: list[str]) -> LadderSpec:
    """Спецификация лестницы exp25: baseline d0, армы из ``tags`` (d1/d2/d3)."""
    key = f"exp25-iam4vp-{{arm}}-{REGION}"
    baseline = key.format(arm="d0")
    arm_order = tuple(key.format(arm=t) for t in tags)
    labels = {baseline: ARM_LABEL["d0"], **{key.format(arm=t): ARM_LABEL[t] for t in tags}}
    return LadderSpec(
        baseline=baseline,
        baseline_label="d0 (без орографии)",
        arm_order=arm_order,
        labels=labels,
        level_arms=arm_order,
        psd_arms=(baseline, *arm_order),
        psd_colors={
            baseline: ARM_COLORS["d0"],
            **{key.format(arm=t): ARM_COLORS[t] for t in tags},
        },
        epoch=EPOCH,
        table_title=f"# Метрики exp25 — маршрут орографии, PI-IAM4VP, USA (эпоха {EPOCH})",
        arm_filename_prefix="exp25-iam4vp-",
    )


def load_by_filename(results_dir: Path) -> tuple[dict, np.ndarray, list[str]]:
    """Сводки армов по ИМЕНИ ФАЙЛА (не по внутреннему ``arm``) + дельты + каналы.

    d1 переиспользует чекпоинт abl16-r3, чьё внутреннее поле ``arm`` —
    ``abl16L-r3-...``; ключ по имени файла (как в paired_deltas.py) держит сводки и
    дельты в одном пространстве имён, иначе render_psd не найдёт арм.
    """
    summaries = {}
    for path in sorted(results_dir.glob("metrics_*.npz")):
        arm = canonical_arm(path.stem.replace("metrics_", ""))
        summaries[arm] = np.load(path, allow_pickle=False)
    deltas = np.load(results_dir / "paired_deltas.npz", allow_pickle=False)
    channels = [str(c) for c in next(iter(summaries.values()))["channels"]]
    return summaries, deltas, channels


def main() -> None:
    """CLI: результаты exp25 → форест/levels/psd/хитмапы в results/iam4vp_usa/."""
    results_dir = HERE / "results" / f"iam4vp_{REGION}"
    heatmaps_dir = results_dir / "heatmaps"
    spec = build_spec(present_arms(results_dir))
    summaries, deltas, channels = load_by_filename(results_dir)

    render_forest(deltas, channels, spec, results_dir / "fig_ci_forest.png")
    for metric in METRICS:
        render_levels(deltas, metric, spec, results_dir / f"fig_levels_{metric}.png")
        metric_dir = heatmaps_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        for arm in spec.arm_order:
            slug = arm.removeprefix(spec.arm_filename_prefix)
            render_arm_metric_heatmap(deltas, arm, metric, spec, metric_dir / f"{slug}.png")
    render_psd(summaries, channels, spec, results_dir / "fig_psd.png")
    write_table(deltas, channels, spec, results_dir / "metrics_table.md")
    print(f"[exp25-figures] {len(summaries)} армов → форест/levels/psd/хитмапы в {results_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
