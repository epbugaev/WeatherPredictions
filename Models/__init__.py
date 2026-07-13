"""Model package: registers active architectures in ``utils.registry.MODELS``.

Importing this package populates the model registry, so YAML configs can
reference active thesis models by string keys without any local imports in
``train.py``. Три семейства: ``"IAM4VP"``/``"PI-IAM4VP"`` (алиасы одного
класса — физика включается параметрами), ``"SimVP"``, ``"PredRNN"``/``"PredRNNv2"``.
"""

from Models.IAM4VP import IAM4VP
from Models.PredRNN import PI_PredRNNv2_Model, PredRNN_Model, PredRNNv2_Model
from Models.SimVP import PI_SimVP_Model, SimVP_Model
from utils.registry import register_model


def _build_simvp(**params):
    """Instantiate ``SimVP_Model``, coercing list ``in_shape`` from YAML to a tuple."""
    if "in_shape" in params:
        params["in_shape"] = tuple(params["in_shape"])
    return SimVP_Model(**params)


def _build_pi_simvp(**params):
    """``PI_SimVP_Model``: SimVPv2 geometry plus the physics-branch params.

    Everything outside the backbone arguments is forwarded to
    ``PhysicsResidualMixin.init_physics_residual`` — the same ``physics_*`` /
    ``diabatic_*`` keys the PI-IAM4VP and PI-PredRNNv2 configs use.
    """
    if "in_shape" in params:
        params["in_shape"] = tuple(params["in_shape"])
    return PI_SimVP_Model(**params)


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


def _build_pi_predrnn_v2(**params):
    """``PI_PredRNNv2_Model``: PredRNNv2 geometry plus the physics-branch params.

    Everything outside ``num_layers``/``num_hidden``/``configs`` is forwarded to
    ``PhysicsResidualMixin.init_physics_residual`` — the same ``physics_*`` /
    ``diabatic_*`` keys the PI-IAM4VP configs use.
    """
    configs = dict(params.pop("configs", {}))
    if "in_shape" in configs:
        configs["in_shape"] = tuple(configs["in_shape"])
    if "num_hidden" in params:
        params["num_hidden"] = tuple(params["num_hidden"])
    return PI_PredRNNv2_Model(configs=configs, **params)


register_model("SimVP")(_build_simvp)
# ``SimVP_Model(model_type="gSTA")`` (дефолт) — это MidMetaNet + gSTA-attention,
# то есть SimVPv2 (v1 — ветка IncepU). Ключ ``SimVPv2`` не врёт о том, что
# обучается; ``SimVP`` оставлен ради старых конфигов.
register_model("SimVPv2")(_build_simvp)
register_model("PI-SimVPv2")(_build_pi_simvp)
# Один класс на семейство: физика включается параметрами конфига
# (use_physics / use_physics_residual_corrector), поэтому оба ключа —
# алиасы одного конструктора.
register_model("IAM4VP")(IAM4VP)
register_model("PI-IAM4VP")(IAM4VP)
register_model("PredRNN")(_build_predrnn)
register_model("PredRNNv2")(_build_predrnn_v2)
register_model("PI-PredRNNv2")(_build_pi_predrnn_v2)
