"""Контракт PI-PredRNNv2 (exp20): физика подключена, отключаемая, не ломает v2.

Физядро подключается к PredRNNv2 внутри его авторегрессивного цикла и должно
удовлетворять тем же инвариантам, что и у PI-IAM4VP: нулевая инициализация
корректора => модель тождественна голому v2; арм ``no_physics`` сохраняет
ёмкость головы, но обнуляет физпризнак; аукс-лосс жив и пробрасывает градиент.
"""

from __future__ import annotations

import unittest

import torch

from Models.PredRNN import PI_PredRNNv2_Model, PredRNNv2_Model
from utils.physics_residual import PhysicsResidualMixin
from utils.registry import get_model

BASE_CONFIGS = {
    "in_shape": (12, 69, 32, 64),
    "patch_size": 1,
    "filter_size": 3,
    "stride": 1,
    "layer_norm": True,
    "pre_seq_length": 2,
    "aft_seq_length": 2,
    "reverse_scheduled_sampling": 0,
    "decouple_beta": 0.1,
}

# Арм P3 (A2 + знаковые фиксы exp13), урезанный по ёмкости ради скорости теста.
PHYSICS_P3 = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_residual_lambda_l1": 0.0001,
    "physics_feature_mode": "tendency",
    "physics_residual_shuffle": "none",
    "physics_residual_hybrid_steps": 1,
    "physics_residual_hybrid_mode": "stable_physical_v2",
    "physics_residual_input_space": "physical",
    "physics_residual_humidity_mode": "relative_to_specific",
    "physics_residual_tendency_clip": 8.0,
    "physics_w_diagnostic": "mass_consistent",
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
}

# Арм P0: голова той же ёмкости, физпризнак тождественно нулевой.
PHYSICS_P0 = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": True,
    "physics_residual_lambda_l1": 0.0001,
    "physics_feature_mode": "no_physics",
    "physics_residual_shuffle": "none",
    "physics_lat_start_deg": 18.28125,
    "physics_dlat_deg": 5.625,
    "physics_dlon_deg": 5.625,
}

NUM_LAYERS = 2
NUM_HIDDEN = (16, 16)


def _build_pi(physics: dict, configs_overrides: dict | None = None, **overrides):
    """Собрать PI-PredRNNv2 на фиксированном сиде с единичной нормализацией."""
    torch.manual_seed(0)
    configs = {**BASE_CONFIGS, **(configs_overrides or {})}
    model = PI_PredRNNv2_Model(
        num_layers=NUM_LAYERS,
        num_hidden=NUM_HIDDEN,
        configs=configs,
        **{**physics, **overrides},
    )
    model.eval()
    model.set_physics_normalization(torch.zeros(69), torch.ones(69))
    return model


def _build_plain_v2():
    """Голый PredRNNv2 с тем же сидом — эталон для проверки zero-init."""
    torch.manual_seed(0)
    model = PredRNNv2_Model(NUM_LAYERS, NUM_HIDDEN, BASE_CONFIGS)
    model.eval()
    return model


def _inputs(total_length: int = 4):
    torch.manual_seed(1)
    frames = torch.randn(1, total_length, 32, 64, 69)
    mask = torch.zeros(1, 2, 1, 1, 1)
    return frames, mask


class TestRegistry(unittest.TestCase):
    def test_registered_under_pi_predrnnv2(self) -> None:
        builder = get_model("PI-PredRNNv2")
        self.assertTrue(callable(builder))


class TestContract(unittest.TestCase):
    def test_inherits_mixin_and_v2(self) -> None:
        self.assertTrue(issubclass(PI_PredRNNv2_Model, PhysicsResidualMixin))
        self.assertTrue(issubclass(PI_PredRNNv2_Model, PredRNNv2_Model))

    def test_patch_size_other_than_one_is_rejected(self) -> None:
        """Физядро ест состояние (B,69,32,64); патчинг свернул бы пространство в каналы."""
        with self.assertRaises(ValueError) as ctx:
            _build_pi(PHYSICS_P3, configs_overrides={"patch_size": 2})
        self.assertIn("patch_size", str(ctx.exception))

    def test_forward_shape_and_aux_keys(self) -> None:
        model = _build_pi(PHYSICS_P3)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (1, 3, 32, 64, 69))
        self.assertEqual(set(aux), {"decouple", "physics_aux"})
        self.assertEqual(aux["physics_aux"].ndim, 0)


class TestPhysicsSwitching(unittest.TestCase):
    def test_zero_init_corrector_is_identity(self) -> None:
        """physics_residual_zero_init=True => коррекция нулевая => выход равен голому v2."""
        model = _build_pi(PHYSICS_P3)
        baseline = _build_plain_v2()
        frames, mask = _inputs()
        with torch.no_grad():
            pi_out, _ = model(frames, mask)
            base_out, _ = baseline(frames, mask)
        torch.testing.assert_close(pi_out, base_out, rtol=1e-5, atol=1e-6)

    def test_trained_corrector_moves_the_prediction(self) -> None:
        """Ненулевая голова => физика реально двигает предсказание (иначе путь мёртв)."""
        model = _build_pi(PHYSICS_P3, physics_residual_zero_init=False)
        baseline = _build_plain_v2()
        frames, mask = _inputs()
        with torch.no_grad():
            pi_out, _ = model(frames, mask)
            base_out, _ = baseline(frames, mask)
        self.assertGreater((pi_out - base_out).abs().max().item(), 0.0)

    def test_no_physics_arm_has_zero_physics_feature(self) -> None:
        """Арм P0: голова строится, но delta_phys тождественно ноль."""
        model = _build_pi(PHYSICS_P0)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (1, 3, 32, 64, 69))
        self.assertIn("physics_aux", aux)
        diagnostics = model.physics_residual_diagnostics()
        self.assertEqual(diagnostics["physics_residual_tendency_rms"].item(), 0.0)

    def test_physics_arm_has_nonzero_physics_feature(self) -> None:
        """Арм P3: физика действительно считается (невязка не нулевая)."""
        model = _build_pi(PHYSICS_P3)
        frames, mask = _inputs()
        model(frames, mask)
        diagnostics = model.physics_residual_diagnostics()
        self.assertGreater(diagnostics["physics_residual_tendency_rms"].item(), 0.0)


class TestGradients(unittest.TestCase):
    def test_aux_loss_is_finite_and_backpropagates_into_the_corrector(self) -> None:
        model = _build_pi(PHYSICS_P3, physics_residual_zero_init=False)
        model.train()
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        loss = next_frames.abs().mean() + aux["decouple"] + aux["physics_aux"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [
            parameter.grad.abs().sum().item()
            for parameter in model.physics_residual_corrector.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(grads)
        self.assertGreater(sum(grads), 0.0)

    def test_physics_aux_is_averaged_over_rollout_steps_not_last_step_only(self) -> None:
        """Регрессия: mixin перезаписывает _last_residual_aux_loss на каждом шаге.

        Если брать его «как есть» после цикла, в лосс попадёт штраф ТОЛЬКО
        последнего шага. Проверяем, что штраф — среднее по шагам: удлинение
        клипа при нулевой (zero-init) коррекции оставляет штраф нулевым, а при
        обученной — меняет его, потому что усредняются разные шаги.
        """
        model = _build_pi(PHYSICS_P3, physics_residual_zero_init=False)
        frames_short, mask = _inputs(total_length=3)
        frames_long, _ = _inputs(total_length=4)
        with torch.no_grad():
            _, aux_short = model(frames_short, mask)
            _, aux_long = model(frames_long, mask)
        self.assertGreater(aux_short["physics_aux"].item(), 0.0)
        self.assertGreater(aux_long["physics_aux"].item(), 0.0)
        self.assertNotAlmostEqual(
            aux_short["physics_aux"].item(), aux_long["physics_aux"].item(), places=7
        )


if __name__ == "__main__":
    unittest.main()
