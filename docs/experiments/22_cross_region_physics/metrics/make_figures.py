"""Фигуры полного набора метрик одной лестницы exp22 (семейство × регион).

exp22 сравнивает 3 арма (`no_physics` / `A2 exp13` / `legacy`) внутри каждой из 6
лестниц (3 семейства × 2 региона). Отрисовка общая с exp16/20/21 и живёт в
:mod:`tools.metrics_ladder`; здесь — только спецификация лестницы (армы, baseline =
`no_physics`, эпоха) и пути. Вход — ``results/<family>_<region>/`` с
``metrics_<arm>.npz`` + ``paired_deltas_<family>_<region>.npz`` (скачаны с кластера,
см. ``sh_files/exp22_metrics_eval.sh``).

Запуск (локально, на лестницу):
    python docs/experiments/22_cross_region_physics/metrics/make_figures.py \
        --family simvpv2 --region npac
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.metrics_ladder import LadderSpec, write_metric_outputs  # noqa: E402

# Общая эпоха у семейства своя (спек §2): PredRNNv2 — 40, остальные — 150.
EPOCH_BY_FAMILY = {"iam4vp": 150, "simvpv2": 150, "predrnnv2": 40}
FAMILY_LABEL = {"iam4vp": "PI-IAM4VP", "simvpv2": "PI-SimVPv2", "predrnnv2": "PI-PredRNNv2"}
REGION_LABEL = {"france": "Европа/Франция", "npac": "Сев. Тихий океан"}
# Okabe-Ito: контроль серый, A2 синий (корректная физика), легаси оранжевый.
ARM_COLORS = {"nophys": "#8a8a8a", "a2": "#0072B2", "legacy": "#E69F00"}


def build_spec(family: str, region: str) -> LadderSpec:
    """Спецификация лестницы (семейство × регион): baseline no_physics, армы A2/legacy."""
    key = f"exp22-{family}-{{arm}}-{region}"
    nophys, a2, legacy = (key.format(arm=a) for a in ("nophys", "a2", "legacy"))
    labels = {
        nophys: "no_physics",
        a2: "A2 exp13 (исправленная)",
        legacy: "legacy (WeatherGFT)",
    }
    return LadderSpec(
        baseline=nophys,
        baseline_label="no_physics",
        arm_order=(a2, legacy),
        labels=labels,
        level_arms=(a2, legacy),
        psd_arms=(nophys, a2, legacy),
        psd_colors={
            k: ARM_COLORS[a] for k, a in ((nophys, "nophys"), (a2, "a2"), (legacy, "legacy"))
        },
        epoch=EPOCH_BY_FAMILY[family],
        table_title=(
            f"# Метрики exp22 — {FAMILY_LABEL[family]}, {REGION_LABEL[region]} "
            f"(эпоха {EPOCH_BY_FAMILY[family]})"
        ),
        arm_filename_prefix=f"exp22-{family}-",
    )


def main() -> None:
    """CLI: (семейство, регион) → форест/levels/psd/хитмапы в results/<family>_<region>/."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["iam4vp", "simvpv2", "predrnnv2"])
    parser.add_argument("--region", required=True, choices=["france", "npac"])
    args = parser.parse_args()
    ladder = f"{args.family}_{args.region}"
    results_dir = HERE / "results" / ladder
    write_metric_outputs(
        spec=build_spec(args.family, args.region),
        results_dir=results_dir,
        figures_dir=results_dir,
        heatmaps_dir=results_dir / "heatmaps",
    )


if __name__ == "__main__":
    main()
