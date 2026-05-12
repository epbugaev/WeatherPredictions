"""Early-stopping tracker driven directly by the trainer loop.

``EarlyStoppingTracker`` is updated once per validation epoch with the
monitored metric value. ``update`` returns ``True`` when the trainer should
stop. The class is rank-aware only through the value it receives — the caller
is responsible for reducing the metric across DDP ranks beforehand so that all
ranks make the same stop decision.
"""

import math


class EarlyStoppingTracker:
    """Monitors a validation metric and signals when to stop training.

    Args:
        patience: Number of consecutive non-improving epochs tolerated.
        mode: ``"min"`` or ``"max"`` — direction of improvement.
        check_finite: If True, stop immediately when the metric is NaN/Inf.
        min_delta: Minimum change in the monitored metric to count as
            improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        check_finite: bool = True,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.check_finite = check_finite
        self.min_delta = min_delta
        self._best = math.inf if mode == "min" else -math.inf
        self._bad_epochs = 0
        self.stopped = False

    @property
    def best(self) -> float:
        """Best metric value seen so far."""
        return self._best

    def is_improvement(self, value: float) -> bool:
        """Whether ``value`` improves on the current best by at least ``min_delta``."""
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def update(self, value: float) -> bool:
        """Feed a new metric value and return whether training should stop.

        Args:
            value: Latest value of the monitored metric.

        Returns:
            ``True`` if training should stop now.
        """
        if self.check_finite and not math.isfinite(value):
            self.stopped = True
            return True

        if self.is_improvement(value):
            self._best = value
            self._bad_epochs = 0
            return False

        self._bad_epochs += 1
        if self._bad_epochs >= self.patience:
            self.stopped = True
            return True
        return False
