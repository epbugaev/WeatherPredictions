"""Экспорт per-step val-метрик abl16-ранов из Comet в results/abl16_metrics.json.

Запуск локально или на кластере (нужен доступ в интернет и Comet-ключ). Ключ и
воркспейс берутся из окружения (``COMET_API_KEY`` / ``COMET_WORKSPACE`` /
``COMET_PROJECT_NAME``); если их нет — подхватываются из ``<repo>/.env`` (тот же
файл, что читает ``sh_files/_shell_contract.sh``).

Схема выхода (потребляется ``make_figures.py``)::

    {
      "meta": {"workspace": ..., "project": ..., "val_every_n_epochs": 3},
      "runs": {
        "abl16-r0-no-physics-s0": {
          "val_loss":      [[step, value], ...],   # отсортировано по step
          "RMSE_z_mean":   [[step, value], ...],
          ...
        },
        ...
      }
    }

Имена метрик не хардкодятся: берётся всё, где в имени есть ``rmse`` или ``val``
(namespace без legacy-префикса, фикс 576983d).
"""

import json
import os
import pathlib

from comet_ml.api import API

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
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


def load_env_file(path: pathlib.Path) -> dict[str, str]:
    """Разобрать ``KEY=value`` из .env (без сторонних зависимостей).

    Args:
        path: путь к .env; если файла нет — возвращается пустой словарь.

    Returns:
        Словарь переменных окружения из файла (кавычки по краям снимаются).
    """
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def resolve_comet_credentials() -> tuple[str, str]:
    """Достать (api_key, workspace) из окружения или ``<repo>/.env``.

    Проект намеренно не резолвится здесь — он фиксирован модульной ``PROJECT``
    (локальный .env может указывать на другой проект).

    Returns:
        Пара (api_key, workspace).
    """
    file_env = load_env_file(REPO_ROOT / ".env")
    api_key = os.environ.get("COMET_API_KEY") or file_env.get("COMET_API_KEY", "")
    workspace = os.environ.get("COMET_WORKSPACE") or file_env.get("COMET_WORKSPACE", "")
    if not api_key or not workspace:
        raise ValueError(
            f"COMET_API_KEY / COMET_WORKSPACE не заданы ни в окружении, ни в {REPO_ROOT / '.env'}"
        )
    return api_key, workspace


def collect_run(experiment) -> dict[str, list[list[float]]]:
    """Собрать все val/rmse-метрики одного эксперимента одним bulk-запросом.

    ``experiment.get_metrics()`` без имени отдаёт все точки всех метрик за один
    вызов (на порядок быстрее, чем per-metric); группируем по ``metricName`` и
    оставляем только val/rmse (namespace без legacy-префикса, фикс 576983d).

    Args:
        experiment: Comet ``APIExperiment``.

    Returns:
        ``{metric_name: [[step, value], ...]}`` — точки отсортированы по step.
    """
    series: dict[str, list[list[float]]] = {}
    for point in experiment.get_metrics():
        name = point.get("metricName")
        if name is None:
            continue
        lowered = name.lower()
        if "rmse" not in lowered and "val" not in lowered:
            continue
        step = point.get("step")
        value = point.get("metricValue")
        if step is None or value is None:
            continue
        series.setdefault(name, []).append([int(step), float(value)])
    for points in series.values():
        points.sort(key=lambda sv: sv[0])
    return series


def main() -> None:
    """Скачать метрики всех abl16-ранов и записать results/abl16_metrics.json."""
    api_key, workspace = resolve_comet_credentials()
    project = PROJECT
    api = API(api_key=api_key)
    runs_out: dict[str, dict] = {}
    for run in RUNS:
        matches = api.get_experiments(workspace, project_name=project, pattern=run)
        exact = [e for e in matches if e.get_name() and e.get_name().startswith(run)]
        if not exact:
            print(f"[collect] MISSING {run}")  # noqa: T201
            continue
        # самый свежий ран с этим именем (суффикс = SLURM job id)
        experiment = sorted(exact, key=lambda e: e.get_name())[-1]
        runs_out[run] = collect_run(experiment)
        n_points = max((len(v) for v in runs_out[run].values()), default=0)
        print(f"[collect] {run}: {len(runs_out[run])} metrics, ≤{n_points} points")  # noqa: T201
    payload = {
        "meta": {
            "workspace": workspace,
            "project": project,
            "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
        },
        "runs": runs_out,
    }
    dst = HERE / "results" / "abl16_metrics.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2))
    print(f"[collect] written {dst} ({len(runs_out)}/{len(RUNS)} runs)")  # noqa: T201


if __name__ == "__main__":
    main()
