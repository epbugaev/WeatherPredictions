"""Экспорт per-epoch val-метрик Comet-ранов в JSON (общее ядро для экспериментов).

Раньше эта логика жила целиком внутри
``docs/experiments/16_model_ablation_ladder/collect_metrics.py``. Эксперимент 20
собирает метрики точно так же — меняются только список ранов, проект и путь
выхода, — поэтому ядро вынесено сюда, а скрипты экспериментов стали тонкими
обёртками (DRY, CLAUDE.md §1).

Схема выхода (её потребляют ``make_figures.py`` обоих экспериментов)::

    {
      "meta": {"workspace": ..., "project": ..., "val_every_n_epochs": 3},
      "runs": {
        "<run-name>": {
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

from __future__ import annotations

import json
import os
import pathlib

from comet_ml.api import API

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


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

    Проект намеренно НЕ резолвится здесь: он фиксируется вызывающим скриптом
    эксперимента (локальный .env может указывать на другой проект).

    Returns:
        Пара (api_key, workspace).

    Raises:
        ValueError: если ключ или воркспейс не найдены ни в окружении, ни в .env.
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
    оставляем только val/rmse.

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


def collect_runs(
    runs: tuple[str, ...],
    project: str,
    destination: pathlib.Path,
    val_every_n_epochs: int,
) -> dict:
    """Скачать метрики всех ``runs`` и записать JSON в ``destination``.

    Args:
        runs: имена Comet-ранов (точное начало имени; суффикс — SLURM job id).
        project: Comet-проект, в котором лежат раны.
        destination: куда писать JSON (родительские каталоги создаются).
        val_every_n_epochs: шаг валидации — пишется в ``meta`` для make_figures.

    Returns:
        Записанный payload.

    Side effects:
        Пишет ``destination`` на диск; печатает прогресс по каждому рану.
    """
    api_key, workspace = resolve_comet_credentials()
    api = API(api_key=api_key)
    runs_out: dict[str, dict] = {}
    for run in runs:
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
            "val_every_n_epochs": val_every_n_epochs,
        },
        "runs": runs_out,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"[collect] written {destination} ({len(runs_out)}/{len(runs)} runs)")  # noqa: T201
    return payload
