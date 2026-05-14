"""Lightweight registries for models, datasets and training strategies.

Each registry is a plain ``dict`` keyed by the string identifier used in YAML
configs (e.g. ``"SimVP"``, ``"v3"``, ``"mutiout_f"``). Components register
themselves via decorators when their module is imported; the top-level
package ``__init__`` modules import every component to populate the registries
before ``train.py`` looks them up.

This module replaces the ``if model_type == ...`` chains with local imports
that previously violated CLAUDE.md ("никаких локальных импортов").
"""

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

MODELS: dict[str, Callable[..., Any]] = {}
DATASETS: dict[str, Callable[..., Any]] = {}
STRATEGIES: dict[str, Callable[..., Any]] = {}


def _register(registry: dict[str, Callable[..., Any]], name: str) -> Callable[[T], T]:
    """Return a decorator that stores ``cls`` in ``registry`` under ``name``.

    Args:
        registry: Target registry dict.
        name: String identifier used in YAML configs.

    Returns:
        Decorator that registers and returns the decorated class unchanged.

    Raises:
        ValueError: if ``name`` is already taken by a different object.
    """

    def decorator(cls: T) -> T:
        if name in registry and registry[name] is not cls:
            raise ValueError(f"Registry collision: {name!r} already bound to {registry[name]!r}")
        registry[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[T], T]:
    """Register a ``nn.Module`` subclass under ``name`` in ``MODELS``."""
    return _register(MODELS, name)


def register_dataset(name: str) -> Callable[[T], T]:
    """Register a ``torch.utils.data.Dataset`` subclass under ``name`` in ``DATASETS``."""
    return _register(DATASETS, name)


def register_strategy(name: str) -> Callable[[T], T]:
    """Register a ``StepStrategy`` subclass under ``name`` in ``STRATEGIES``."""
    return _register(STRATEGIES, name)


def get_model(name: str) -> Callable[..., Any]:
    """Look up a model factory; raises KeyError with a helpful message."""
    if name not in MODELS:
        raise KeyError(f"Unknown model type {name!r}. Available: {sorted(MODELS)}")
    return MODELS[name]


def get_dataset(name: str) -> Callable[..., Any]:
    """Look up a dataset factory; raises KeyError with a helpful message."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset version {name!r}. Available: {sorted(DATASETS)}")
    return DATASETS[name]


def get_strategy(name: str) -> Callable[..., Any]:
    """Look up a strategy factory; raises KeyError with a helpful message."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
