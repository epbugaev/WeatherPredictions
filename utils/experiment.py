"""Experiment-logging facade for Comet ML.

``CometExperiment`` wraps the official ``comet_ml.Experiment`` with the small
API surface the trainer actually needs (``log_metrics``, ``log_figure``,
``log_code``, ``close``). ``NullExperiment`` is a no-op stand-in used on
non-main DDP ranks and in tests.

The trainer always programs against the ``Experiment`` protocol, so log
calls in strategies do not need to know whether they're on rank-0.
"""

from __future__ import annotations

import os
import random
import string
from typing import Any, Protocol

import comet_ml
import matplotlib.figure


class Experiment(Protocol):
    """Minimal protocol used by the trainer and strategies for logging."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_figure(
        self,
        figure: matplotlib.figure.Figure,
        name: str,
        step: int | None = None,
    ) -> None: ...
    def log_code(self, file_name: str) -> None: ...
    def close(self) -> None: ...


class NullExperiment:
    """No-op experiment used on non-main ranks and when logging is disabled."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return

    def log_figure(
        self,
        figure: matplotlib.figure.Figure,
        name: str,
        step: int | None = None,
    ) -> None:
        return

    def log_code(self, file_name: str) -> None:
        return

    def close(self) -> None:
        return


class CometExperiment:
    """Thin wrapper around ``comet_ml.Experiment``.

    Args:
        project_name: Comet project name.
        experiment_name: Human-readable base name; a run-unique suffix
            (``$SLURM_JOB_ID`` on the cluster, else a random token) is
            appended so re-runs of one config stay distinguishable in Comet.
        api_key: Comet API key; if empty, an environment-variable lookup is
            performed inside ``comet_ml``.
    """

    def __init__(self, project_name: str, experiment_name: str, api_key: str = "") -> None:
        if api_key:
            os.environ["COMET_API_KEY"] = api_key
        run_key = "".join(random.choices(string.ascii_lowercase + string.digits, k=50))
        os.environ["COMET_EXPERIMENT_KEY"] = run_key
        comet_ml.login()
        self._experiment = comet_ml.Experiment(project_name=project_name)
        # Comet does not enforce unique display names: re-running the same
        # YAML config (e.g. round-1 vs round-2, or after a fix) would create
        # several experiments all literally named ``experiment_name`` and they
        # collapse into indistinguishable rows on comparison panels. Append a
        # run-unique suffix — the SLURM job id when launched on the cluster
        # (also ties the Comet run to its sbatch job), else the random run key.
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        run_suffix = slurm_job_id if slurm_job_id else run_key[:6]
        self._experiment.set_name(f"{experiment_name}-{run_suffix}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Forward a scalar metrics dict to Comet."""
        self._experiment.log_metrics(metrics, step=step)

    def log_figure(
        self,
        figure: matplotlib.figure.Figure,
        name: str,
        step: int | None = None,
    ) -> None:
        """Log a matplotlib figure under the given name."""
        self._experiment.log_figure(figure=figure, figure_name=name, step=step)

    def log_code(self, file_name: str) -> None:
        """Upload a source file as a code artifact."""
        self._experiment.log_code(file_name=file_name)

    def close(self) -> None:
        """End the Comet experiment so the SDK uploads pending data."""
        self._experiment.end()


def build_experiment(
    logging_cfg: dict[str, Any],
    experiment_name: str,
    is_main: bool,
) -> Experiment:
    """Build the right experiment for this rank.

    Args:
        logging_cfg: ``logging`` section from the YAML config; reads
            ``comet_project``. The Comet API key is taken from
            ``$COMET_API_KEY`` (see ``.env`` and ``sh_files/_shell_contract.sh``);
            ``logging.comet_api_key`` is read only as a legacy override
            and should be left unset in committed configs.
        experiment_name: Display name (taken from ``experiment.name`` in YAML).
        is_main: Only main-rank processes create a real Comet experiment.

    Returns:
        ``CometExperiment`` on the main rank, ``NullExperiment`` elsewhere.
    """
    if not is_main:
        return NullExperiment()

    project_name = logging_cfg.get("comet_project", "WeatherPredictions")
    api_key = logging_cfg.get("comet_api_key", "")
    return CometExperiment(
        project_name=project_name,
        experiment_name=experiment_name,
        api_key=api_key,
    )
