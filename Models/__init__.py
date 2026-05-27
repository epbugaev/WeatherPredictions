"""Model package: registers every architecture in ``utils.registry.MODELS``.

Importing this package populates the model registry, so YAML configs can
reference models by string keys like ``"SimVP"`` or ``"PredRNN"`` without
any local imports in ``train.py``.
"""

from Models.PI_IAM4VP import IAM4VP
from Models.PredFormer import PredFormer_Model as PredFormer_Model_v1
from Models.PredFormerGFT import PredFormer_Model as PredFormer_Model_GFT
from Models.PredFormerGFT_HybridBlock import PredFormer_Model as PredFormer_Model_GFT_Hybrid
from Models.PredRNN import PredRNN_Model, PredRNNv2_Model
from Models.SimVP import SimVP_Model
from Models.WeatherGFT import GFT as WeatherGFT
from Models.WeatherGFTSingle import GFT as WeatherGFTSingle
from utils.registry import register_model


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


def _build_predformer_gft_hybrid(**params):
    """``PredFormerGFT_HybridBlock`` takes the params dict positionally."""
    return PredFormer_Model_GFT_Hybrid(params)


def _build_predrnn(**params):
    """``PredRNN_Model`` with nested ``configs`` and tuple coercion."""
    configs = dict(params.pop("configs", {}))
    if "in_shape" in configs:
        configs["in_shape"] = tuple(configs["in_shape"])
    if "num_hidden" in params:
        params["num_hidden"] = tuple(params["num_hidden"])
    return PredRNN_Model(configs=configs, **params)


def _build_predrnn_v2(**params):
    """``PredRNNv2_Model`` with nested ``configs`` and tuple coercion."""
    configs = dict(params.pop("configs", {}))
    if "in_shape" in configs:
        configs["in_shape"] = tuple(configs["in_shape"])
    if "num_hidden" in params:
        params["num_hidden"] = tuple(params["num_hidden"])
    return PredRNNv2_Model(configs=configs, **params)


register_model("SimVP")(_build_simvp)
register_model("WeatherGFT")(WeatherGFT)
register_model("WeatherGFTSingle")(WeatherGFTSingle)
register_model("PredFormer")(_build_predformer)
register_model("PredFormerGFT")(_build_predformer_gft)
register_model("PredFormerGFT_HybridBlock")(_build_predformer_gft_hybrid)
register_model("PI-IAM4VP")(IAM4VP)
register_model("PredRNN")(_build_predrnn)
register_model("PredRNNv2")(_build_predrnn_v2)
