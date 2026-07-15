"""Экспорт per-epoch val-метрик exp21-ранов из Comet в results/exp21_metrics.json.

Логика скачивания общая с exp16 и exp20, живёт в :mod:`tools.comet_collect`;
здесь — только специфика exp21: список арм лестницы, проект и путь выхода.
Берётся long-волна (t=12, 500 эпох) — именно на ней делаются выводы (§8.5 README).

Запуск::

    python docs/experiments/21_pi_simvpv2_ladder/collect_metrics.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.comet_collect import collect_runs  # noqa: E402

# Валидация каждые 10 эпох (configs/exp21_long/*.yaml, trainer.val_every_n_epochs).
VAL_EVERY_N_EPOCHS = 10
# Тот же Comet-проект, что у exp16/exp20 — чтобы все лестницы лежали рядом.
PROJECT = "pi-iamvp"

# Порядок = порядок лестницы (§4 README): baseline, легаси, A-семейство (batched),
# затем контроль связки chained (S0c без физики, S3c с физикой).
RUNS = (
    "exp21L-s0-no-physics-t12-seed0",
    "exp21L-s1-legacy-hybrid-t12-seed0",
    "exp21L-s3a-no-diabatic-t12-seed0",
    "exp21L-s3-a2-exp13-t12-seed0",
    "exp21L-s4-exp14-t12-seed0",
    "exp21L-s5-exp15-t12-seed0",
    "exp21L-s0c-no-physics-chained-t12-seed0",
    "exp21L-s3c-a2-exp13-chained-t12-seed0",
)


def main() -> None:
    """Скачать метрики всех exp21-ранов и записать results/exp21_metrics.json."""
    collect_runs(
        runs=RUNS,
        project=PROJECT,
        destination=HERE / "results" / "exp21_metrics.json",
        val_every_n_epochs=VAL_EVERY_N_EPOCHS,
    )


if __name__ == "__main__":
    main()
