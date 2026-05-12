"""Model package: registers every architecture in ``utils.registry.MODELS``.

Importing this package populates the model registry, so YAML configs can
reference models by string keys like ``"SimVP"`` or ``"PredRNN"`` without
any local imports in ``train.py``.

A handful of legacy entries (e.g. ``PredRNN``) point at modules that have
since been removed from the repository; we register those only when the
backing module actually exists (LBYL via :func:`importlib.util.find_spec`)
to keep the registry usable in the common case while honouring the request
to keep all training scripts in tree.
"""

import importlib
import importlib.util
from typing import Any

from Models.PI_IAM4VP import IAM4VP
from Models.PredFormer import PredFormer_Model as PredFormer_Model_v1
from Models.PredFormerGFT import PredFormer_Model as PredFormer_Model_GFT
from Models.SimVP import SimVP_Model
from Models.WeatherGFT import GFT as WeatherGFT
from Models.WeatherGFTSingle import GFT as WeatherGFTSingle
from utils.registry import register_model


def _maybe_import(module_path: str, attr: str) -> Any | None:
    """Return ``module.attr`` if ``module_path`` resolves, else ``None``.

    LBYL alternative to ``try: import ... except ImportError``; used to make
    registrations of legacy models tolerate missing files without violating
    the project rule against ``try/except``.
    """
    if importlib.util.find_spec(module_path) is None:
        return None
    return getattr(importlib.import_module(module_path), attr)


def _build_simvp(**params):
    """Instantiate ``SimVP_Model``, coercing list ``in_shape`` from YAML to a tuple."""
    if "in_shape" in params:
        params["in_shape"] = tuple(params["in_shape"])
    return SimVP_Model(**params)


def _build_predformer(**params):
    """``PredFormer_Model`` takes the params dict positionally (not unpacked)."""
    return PredFormer_Model_v1(params)


def _build_predformer_gft(**params):
    """``PredFormerGFT_Model`` takes the params dict positionally (not unpacked)."""
    return PredFormer_Model_GFT(params)


register_model("SimVP")(_build_simvp)
register_model("WeatherGFT")(WeatherGFT)
register_model("WeatherGFTSingle")(WeatherGFTSingle)
register_model("PredFormer")(_build_predformer)
register_model("PredFormerGFT")(_build_predformer_gft)
register_model("PI-IAM4VP")(IAM4VP)


_PredRNN_Model = _maybe_import("Models.PredRNN", "PredRNN_Model")
if _PredRNN_Model is not None:
    def _build_predrnn(**params):
        """``PredRNN_Model`` with nested ``configs`` and tuple coercion."""
        configs = dict(params.pop("configs", {}))
        if "in_shape" in configs:
            configs["in_shape"] = tuple(configs["in_shape"])
        if "num_hidden" in params:
            params["num_hidden"] = tuple(params["num_hidden"])
        return _PredRNN_Model(configs=configs, **params)

    register_model("PredRNN")(_build_predrnn)
