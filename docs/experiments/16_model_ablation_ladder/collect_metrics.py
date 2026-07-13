"""Экспорт per-step val-метрик abl16-ранов из Comet в results/abl16_metrics.json.

Запуск локально или на кластере (нужен доступ в интернет и Comet-ключ). Ключ и
воркспейс берутся из окружения (``COMET_API_KEY`` / ``COMET_WORKSPACE``); если их
нет — подхватываются из ``<repo>/.env`` (тот же файл, что читает
``sh_files/_shell_contract.sh``).

Логика скачивания — общая с эксперимента 20, вынесена в
:mod:`tools.comet_collect`; здесь остаётся только то, что специфично для exp16:
список ранов, проект и путь выхода.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.comet_collect import collect_runs  # noqa: E402

VAL_EVERY_N_EPOCHS = 3
# exp16-раны логируются в этот Comet-проект (кластерный .env COMET_PROJECT_NAME=pi-iamvp).
# Проект фиксирован наравне со списком RUNS — локальный .env может указывать на другой
# проект (напр. weatherpredictions), поэтому из env его НЕ читаем.
PROJECT = "pi-iamvp"

RUNS = (
    "abl16-r0-no-physics-s0",
    "abl16-r1-legacy-hybrid-s0",
    "abl16-r2a-a1-pre13-s0",
    "abl16-r2-a2-pre13-s0",
    "abl16-r3-a2-exp13-s0",
    "abl16-r4-exp14-s0",
    "abl16-r5-exp15-s0",
)


def main() -> None:
    """Скачать метрики всех abl16-ранов и записать results/abl16_metrics.json."""
    collect_runs(
        runs=RUNS,
        project=PROJECT,
        destination=HERE / "results" / "abl16_metrics.json",
        val_every_n_epochs=VAL_EVERY_N_EPOCHS,
    )


if __name__ == "__main__":
    main()
