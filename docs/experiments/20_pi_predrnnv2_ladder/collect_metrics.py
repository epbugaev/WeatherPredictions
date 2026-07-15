"""Экспорт per-epoch val-метрик exp20-ранов из Comet в results/exp20_metrics.json.

Логика скачивания общая с exp16 и живёт в :mod:`tools.comet_collect`; здесь —
только специфика exp20: список арм лестницы, проект и путь выхода.

Запуск::

    .venv-exp20/bin/python docs/experiments/20_pi_predrnnv2_ladder/collect_metrics.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.comet_collect import collect_runs  # noqa: E402

# Валидация каждые 2 эпохи (configs/exp20/*.yaml, trainer.val_every_n_epochs).
VAL_EVERY_N_EPOCHS = 2
# Тот же Comet-проект, что у exp16 — чтобы обе лестницы лежали рядом.
PROJECT = "pi-iamvp"

# Порядок = порядок лестницы (§4 README): контроль, легаси, затем A-семейство.
RUNS = (
    "exp20-p0-no-physics-s0",
    "exp20-p1-legacy-hybrid-s0",
    "exp20-p3a-no-diabatic-s0",
    "exp20-p3-a2-exp13-s0",
    "exp20-p4-exp14-s0",
    "exp20-p5-exp15-s0",
)


def main() -> None:
    """Скачать метрики всех exp20-ранов и записать results/exp20_metrics.json."""
    collect_runs(
        runs=RUNS,
        project=PROJECT,
        destination=HERE / "results" / "exp20_metrics.json",
        val_every_n_epochs=VAL_EVERY_N_EPOCHS,
    )


if __name__ == "__main__":
    main()
