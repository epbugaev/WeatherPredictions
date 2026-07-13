"""Контракт PI-SimVPv2 (exp21): физика на MIMO-носителе, две связки prev.

SimVP — MIMO: все T кадров выдаются одним forward, рекуррентности нет. Физядро
же по природе парное (state(t) -> state(t+1)), поэтому «предыдущее состояние»
для кадра t приходится сконструировать. Отсюда два режима ``physics_coupling``:

* ``batched``  — prev = сдвиг [x[-1], y_nn[0..T-2]] (прогноз бэкбона), один
  вызов физядра на батче B*T; MIMO-характер модели сохраняется;
* ``chained`` — prev = скорректированный кадр t-1 (семантика PI-IAM4VP), T
  последовательных вызовов; арм S3c, измеряющий цену авторегрессивной связки.
"""

from __future__ import annotations

import unittest

import torch

from Models.SimVP import PI_SimVP_Model, SimVP_Model
from utils.physics_residual import PhysicsResidualMixin
from utils.registry import get_model

IN_SHAPE = (4, 69, 32, 64)

PHYSICS_S3 = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_residual_lambda_l1": 1e-4,
    "physics_feature_mode": "tendency",
    "physics_residual_hybrid_steps": 1,
    "physics_residual_hybrid_mode": "stable_physical_v2",
    "physics_residual_input_space": "physical",
    "physics_residual_humidity_mode": "relative_to_specific",
    "physics_w_diagnostic": "mass_consistent",
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
}

PHYSICS_S0 = {**PHYSICS_S3, "physics_feature_mode": "no_physics"}


def _build(physics: dict, **overrides) -> PI_SimVP_Model:
    torch.manual_seed(0)
    params = {**physics, **overrides}
    model = PI_SimVP_Model(in_shape=IN_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2, **params)
    model.eval()
    model.set_physics_normalization(torch.zeros(69), torch.ones(69))
    return model


def _backbone() -> SimVP_Model:
    torch.manual_seed(0)
    model = SimVP_Model(in_shape=IN_SHAPE, hid_S=8, hid_T=32, N_S=4, N_T=2)
    model.eval()
    return model


def _clip() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, IN_SHAPE[0], 69, 32, 64)


class TestRegistry(unittest.TestCase):
    def test_simvpv2_key_exists(self) -> None:
        """SimVP_Model с model_type=gSTA — это и есть SimVPv2; имя не должно врать."""
        model = get_model("SimVPv2")(in_shape=[4, 69, 32, 64], hid_S=8, hid_T=32, N_S=4, N_T=2)
        self.assertIsInstance(model, SimVP_Model)

    def test_pi_simvpv2_key_builds_the_pi_model(self) -> None:
        model = get_model("PI-SimVPv2")(
            in_shape=[4, 69, 32, 64], hid_S=8, hid_T=32, N_S=4, N_T=2, **PHYSICS_S3
        )
        self.assertIsInstance(model, PI_SimVP_Model)


class TestContract(unittest.TestCase):
    def test_inherits_mixin_and_backbone(self) -> None:
        self.assertTrue(issubclass(PI_SimVP_Model, PhysicsResidualMixin))
        self.assertTrue(issubclass(PI_SimVP_Model, SimVP_Model))

    def test_unknown_coupling_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _build(PHYSICS_S3, physics_coupling="autoregressive")
        self.assertIn("physics_coupling", str(ctx.exception))

    def test_forward_keeps_the_mimo_shape(self) -> None:
        for coupling in ("batched", "chained"):
            with self.subTest(coupling=coupling):
                model = _build(PHYSICS_S3, physics_coupling=coupling)
                out = model(_clip())
                self.assertEqual(out.shape, (2, 4, 69, 32, 64))

    def test_zero_init_corrector_is_identity(self) -> None:
        """physics_residual_zero_init=True => модель стартует как чистый SimVPv2."""
        for coupling in ("batched", "chained"):
            with self.subTest(coupling=coupling):
                model = _build(PHYSICS_S3, physics_coupling=coupling)
                with torch.no_grad():
                    pi_out = model(_clip())
                    base_out = _backbone()(_clip())
                torch.testing.assert_close(pi_out, base_out, rtol=1e-5, atol=1e-6)

    def test_trained_head_moves_the_prediction(self) -> None:
        model = _build(PHYSICS_S3, physics_residual_zero_init=False)
        with torch.no_grad():
            pi_out = model(_clip())
            base_out = _backbone()(_clip())
        self.assertGreater((pi_out - base_out).abs().max().item(), 0.0)

    def test_batched_and_chained_differ_once_the_head_is_nonzero(self) -> None:
        """Оба режима — не одно и то же: prev у них разный (арм S3c имеет смысл)."""
        batched = _build(PHYSICS_S3, physics_coupling="batched", physics_residual_zero_init=False)
        chained = _build(PHYSICS_S3, physics_coupling="chained", physics_residual_zero_init=False)
        with torch.no_grad():
            out_batched = batched(_clip())
            out_chained = chained(_clip())
        # Кадр 0 у обоих режимов строится от одного prev = x[:, -1] -> совпадает.
        torch.testing.assert_close(out_batched[:, 0], out_chained[:, 0], rtol=1e-5, atol=1e-6)
        # Начиная с кадра 1 prev расходится (прогноз бэкбона vs скорректированный).
        self.assertGreater((out_batched[:, 1:] - out_chained[:, 1:]).abs().max().item(), 0.0)

    def test_no_physics_arm_runs_and_keeps_the_head(self) -> None:
        """S0: голова той же ёмкости, но физпризнак тождественно нулевой."""
        model = _build(PHYSICS_S0, physics_residual_zero_init=False)
        out = model(_clip())
        self.assertEqual(out.shape, (2, 4, 69, 32, 64))
        self.assertIsNotNone(model.physics_residual_corrector)

    def test_aux_loss_is_finite_and_reaches_the_head(self) -> None:
        for coupling in ("batched", "chained"):
            with self.subTest(coupling=coupling):
                model = _build(
                    PHYSICS_S3, physics_coupling=coupling, physics_residual_zero_init=False
                )
                out = model(_clip())
                aux = model.physics_residual_aux_loss()
                self.assertTrue(torch.isfinite(aux))
                (out.abs().mean() + aux).backward()
                grads = [
                    p.grad.abs().sum().item()
                    for p in model.physics_residual_corrector.parameters()
                    if p.grad is not None
                ]
                self.assertTrue(grads and sum(grads) > 0.0)

    def test_chunking_does_not_change_the_result(self) -> None:
        """Физпуть посэмплово независим => размер чанка не влияет на выход.

        Чанк существует только ради лимита CUDA-грида: WENO-производная
        разворачивает латент в (B*13*16, H) и зовёт reflection_pad1d, чей кернел
        кладёт плоский батч в grid-измерение с пределом 65535. При B*T = 768
        (batch 64 x 12 кадров) это 159 744 -> "CUDA error: invalid configuration
        argument" (смоук-джоба 4175599).
        """
        whole = _build(PHYSICS_S3, physics_residual_zero_init=False, physics_chunk_size=1000)
        chunked = _build(PHYSICS_S3, physics_residual_zero_init=False, physics_chunk_size=3)
        with torch.no_grad():
            out_whole = whole(_clip())
            out_chunked = chunked(_clip())
        torch.testing.assert_close(out_chunked, out_whole, rtol=0, atol=0)
        torch.testing.assert_close(
            chunked.physics_residual_aux_loss(),
            whole.physics_residual_aux_loss(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_nonpositive_chunk_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _build(PHYSICS_S3, physics_chunk_size=0)
        self.assertIn("physics_chunk_size", str(ctx.exception))

    def test_physics_stays_finite(self) -> None:
        model = _build(PHYSICS_S3, physics_residual_zero_init=False)
        out = model(_clip())
        self.assertTrue(torch.isfinite(out).all())
        diagnostics = model.physics_residual_diagnostics()
        self.assertEqual(diagnostics["physics_residual_nonfinite_ratio"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
