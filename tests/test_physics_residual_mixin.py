"""Побитовая эквивалентность IAM4VP до и после выноса физики в миксин.

Эталон (``tests/goldens/iam4vp_physics_residual_r5.pt``) снят с кода ДО
рефакторинга. Если тест падает — рефакторинг изменил поведение модели, на
которой стоят результаты exp16; чинить надо рефакторинг, а не переснимать
эталон.
"""

from __future__ import annotations

import pathlib
import unittest
import warnings

import torch

from Models.IAM4VP import IAM4VP
from utils.physics_residual import PhysicsResidualMixin

GOLDEN_PATH = pathlib.Path(__file__).parent / "goldens" / "iam4vp_physics_residual_r5.pt"

PHYSICS_ARM_R5 = {
    "use_physics": False,
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 32,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_feature_mode": "tendency",
    "physics_residual_hybrid_steps": 1,
    "physics_residual_hybrid_mode": "stable_physical_v2",
    "physics_w_diagnostic": "mass_consistent",
    "physics_advection_form": "advective",
    "physics_metric_terms": True,
    "physics_spherical_divergence": True,
    "physics_rayleigh_friction": True,
    "physics_vertical_scheme": "lagrange3",
    "physics_omega_free": ["t", "q", "u", "v"],
    "physics_latent_heating_coupling": True,
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
    "T_data": 12,
}

PHYSICS_ARM_R5_TRAINED = {
    **PHYSICS_ARM_R5,
    "physics_residual_zero_init": False,
    "physics_residual_lambda_l1": 1e-4,
}

PHYSICS_ARM_R1_LEGACY = {
    "use_physics": False,
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 32,
    "physics_residual_apply_to": "all_channels",
    "physics_residual_zero_init": False,
    "physics_residual_lambda_l1": 1e-4,
    "physics_feature_mode": "prior_and_tendency",
    "physics_residual_hybrid_steps": 2,
    "physics_residual_hybrid_mode": "legacy_normalized",
    "physics_t_t_formulation": "legacy_paper",
    "physics_use_universal_R": True,
    "physics_coriolis_formulation": "beta_plane",
    "physics_tendency_limiter": "scale_diff",
    "physics_tendency_on_latent": False,
    "T_data": 12,
}

ARMS = {
    "r5_zero_init": PHYSICS_ARM_R5,
    "r5_trained": PHYSICS_ARM_R5_TRAINED,
    "r1_legacy": PHYSICS_ARM_R1_LEGACY,
}

# Методы, которые обязаны приходить из миксина, а не остаться копией в IAM4VP.
MOVED_METHODS = (
    "set_physics_normalization",
    "_require_physics_normalization",
    "_denormalize_state",
    "_normalize_state",
    "_nonfinite_ratio",
    "_finite_or_fallback",
    "_finite_clamp",
    "_sanitize_hybrid_param_grad",
    "_sanitize_physical_parts",
    "_sanitize_hybrid_latent_physical",
    "_clip_normalized_tendency",
    "_hybrid_block_forward",
    "_hybrid_bn_gamma_drift",
    "x_to_zquvtw",
    "_rms",
    "_load_static_geo",
    "_physics_prior_from_state",
    "_build_diabatic_mask",
    "_residual_slice",
    "_apply_physics_residual",
    "physics_residual_aux_loss",
    "physics_residual_diagnostics",
    "set_residual_warmup",
)


def _build_iam4vp(arm: dict[str, object], **overrides: object) -> IAM4VP:
    """Build an eval-mode IAM4VP on the fixed golden seed and normalization.

    Args:
        arm: the complete constructor kwargs of one ladder arm. Passed whole —
            arms must NOT be layered on top of each other, or a key one arm
            leaves at its default would silently inherit another arm's value.
        **overrides: extra constructor kwargs on top of ``arm``.

    Returns:
        IAM4VP in eval mode with physics normalization already installed.
    """
    torch.manual_seed(0)
    params = dict(arm)
    params.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = IAM4VP(**params)
    model.eval()
    model.set_physics_normalization(
        torch.linspace(-1.0, 1.0, 69) * 10.0 + 250.0, torch.linspace(1.0, 3.0, 69)
    )
    return model


def _golden_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(prev_state, y_nn)`` pair the golden was snapshotted on.

    Returns:
        Tuple of two ``(2, 69, 32, 64)`` normalized tensors.
    """
    torch.manual_seed(1)
    prev_state = torch.randn(2, 69, 32, 64)
    y_nn = torch.randn(2, 69, 32, 64)
    return prev_state, y_nn


class TestIAM4VPInheritsMixin(unittest.TestCase):
    """IAM4VP получает физический путь из миксина, а не хранит свою копию."""

    def test_iam4vp_is_a_physics_residual_mixin(self) -> None:
        self.assertTrue(issubclass(IAM4VP, PhysicsResidualMixin))

    def test_mixin_is_not_an_nn_module(self) -> None:
        """Миксин полагается на register_buffer/parameters() носителя."""
        self.assertNotIn(torch.nn.Module, PhysicsResidualMixin.__mro__)

    def test_moved_methods_are_not_redefined_on_iam4vp(self) -> None:
        for name in MOVED_METHODS:
            self.assertNotIn(name, IAM4VP.__dict__, f"{name} must come from the mixin")
            self.assertIn(name, PhysicsResidualMixin.__dict__, f"{name} must live in the mixin")

    def test_duck_typed_attributes_survive(self) -> None:
        """iterative_manual.py:72-101 читает эти имена через getattr."""
        model = _build_iam4vp(
            PHYSICS_ARM_R5, freeze_iam4vp_for_residual_warmup=True, residual_warmup_epochs=2
        )
        self.assertTrue(model.freeze_iam4vp_for_residual_warmup)
        self.assertEqual(model.residual_warmup_epochs, 2)
        model.set_residual_warmup(True)
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen)
        self.assertTrue(all(not n.startswith("physics_residual_corrector") for n in frozen))
        model.set_residual_warmup(False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class TestPhysicsResidualGolden(unittest.TestCase):
    """Выход физического пути не изменился при выносе в миксин."""

    def setUp(self) -> None:
        self.assertTrue(GOLDEN_PATH.exists(), f"missing golden: {GOLDEN_PATH}")
        self.golden = torch.load(GOLDEN_PATH, weights_only=True)

    def _assert_arm_matches_golden(self, arm: str) -> None:
        model = _build_iam4vp(ARMS[arm])
        prev_state, y_nn = _golden_inputs()
        with torch.no_grad():
            prior = model._physics_prior_from_state(prev_state)
            y_hat = model._apply_physics_residual(y_nn, prev_state)

        expected = self.golden[arm]
        torch.testing.assert_close(y_hat, expected["y_hat"], rtol=0, atol=0)
        torch.testing.assert_close(
            model.physics_residual_aux_loss(), expected["aux_loss"], rtol=0, atol=0
        )
        if "prior" in expected:
            torch.testing.assert_close(prior, expected["prior"], rtol=0, atol=0)

        diagnostics = model.physics_residual_diagnostics()
        self.assertEqual(sorted(diagnostics), sorted(expected["diagnostics"]))
        for key, value in expected["diagnostics"].items():
            torch.testing.assert_close(diagnostics[key], value, rtol=0, atol=0, msg=key)

    def test_r5_zero_init_matches_golden(self) -> None:
        """R5 (exp15-полная лестница), zero-init голова: пин физприора."""
        self._assert_arm_matches_golden("r5_zero_init")

    def test_r5_trained_matches_golden(self) -> None:
        """Ненулевая голова: физика реально доезжает до выхода."""
        self._assert_arm_matches_golden("r5_trained")

    def test_r1_legacy_matches_golden(self) -> None:
        """Легаси-ветка (legacy_normalized, all_channels, prior_and_tendency)."""
        self._assert_arm_matches_golden("r1_legacy")


if __name__ == "__main__":
    unittest.main()
