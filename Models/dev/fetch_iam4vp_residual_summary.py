"""Fetch a compact Comet comparison table for IAM4VP residual experiments.

Usage on the cluster after runs finish:

    python Models/dev/fetch_iam4vp_residual_summary.py

The script reads `.env` if present and expects `COMET_API_KEY`; set
`COMET_WORKSPACE` and optionally `COMET_PROJECT_NAME` there too.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from comet_ml.api import API  # noqa: E402


DEFAULT_EXPERIMENT_NAMES = (
    "IAM4VP-USA-v4",
    "PI-IAM4VP-USA-v4",
    "PI-IAM4VP-ResidualNoPhysics-USA-v4",
    "PI-IAM4VP-Residual-USA-v4",
    "PI-IAM4VP-ResidualShuffledPhysics-USA-v4",
)

DEFAULT_METRICS = (
    "val_loss",
    "RMSE_z500_mean",
    "RMSE_t500_mean",
    "RMSE_u500_mean",
    "RMSE_v500_mean",
    "val_physics_residual_correction_rms",
    "val_physics_residual_tendency_rms",
    "val_physics_residual_correction_to_prediction_ratio",
)


def _latest_metric(experiment, metric_name: str) -> float | None:
    metrics = experiment.get_metrics(metric_name) or []
    if not metrics:
        return None
    metrics = sorted(metrics, key=lambda item: int(item.get("step") or 0))
    try:
        return float(metrics[-1]["metricValue"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value != value:
        return "nan"
    return f"{value:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        nargs="*",
        default=list(DEFAULT_EXPERIMENT_NAMES),
        help="Comet experiment names to compare; newest run for each name is used.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metric names to print.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("COMET_API_KEY")
    workspace = os.environ.get("COMET_WORKSPACE")
    project = os.environ.get("COMET_PROJECT_NAME", "weatherpredictions")
    if not api_key:
        raise SystemExit("Set COMET_API_KEY in .env or the environment.")
    if not workspace:
        raise SystemExit("Set COMET_WORKSPACE in .env or the environment.")

    api = API(api_key=api_key)
    experiments = api.get_experiments(workspace=workspace, project_name=project)
    wanted = set(args.names)
    newest_by_name = {}
    for experiment in sorted(
        experiments,
        key=lambda item: item.start_server_timestamp or 0,
        reverse=True,
    ):
        name = experiment.get_name() or experiment.id
        if name in wanted and name not in newest_by_name:
            newest_by_name[name] = experiment

    header = ["experiment", *args.metrics, "url"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for name in args.names:
        experiment = newest_by_name.get(name)
        if experiment is None:
            values = ["missing", *["-"] * len(args.metrics), "-"]
        else:
            values = [
                "ok",
                *[_fmt(_latest_metric(experiment, metric)) for metric in args.metrics],
                f"https://www.comet.com/{workspace}/{project}/{experiment.id}",
            ]
        print("| " + " | ".join([name, *values]) + " |")


if __name__ == "__main__":
    main()
