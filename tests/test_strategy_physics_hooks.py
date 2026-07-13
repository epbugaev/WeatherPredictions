"""Физические хуки живут в StepStrategy, и SimpleStep их использует (exp21).

До exp21 хелперы чтения физ-аукс-лосса и физ-диагностики были продублированы
в ``iterative_manual.py`` и ``predrnn.py``, а ``SimpleStep`` (стратегия SimVP)
их вовсе не имел. Для PI-SimVPv2 это означало бы, что L1-штраф на коррекцию не
доходит до оптимизатора, а ``physics_residual_nonfinite_ratio`` не попадает в
Comet — то есть тихая деградация физветки в fallback была бы невидима.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from training_strategies.base import StepContext, StepStrategy
from training_strategies.simple import SimpleStep
from utils.metrics import Metrics


class _PlainModel(nn.Module):
    """Модель без физики: физ-хуки должны молчать."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _PhysicsModel(_PlainModel):
    """Duck-typed PI-модель: тот же контракт, что у PhysicsResidualMixin."""

    def physics_residual_aux_loss(self) -> torch.Tensor:
        return self.weight.abs().sum() * 0.5

    def physics_residual_diagnostics(self) -> dict[str, torch.Tensor]:
        return {"physics_residual_nonfinite_ratio": torch.zeros(())}


def _context() -> StepContext:
    model = _PlainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mean = torch.zeros(69)
    std = torch.ones(69)
    return StepContext(
        device=torch.device("cpu"),
        optimizer=optimizer,
        scaler=None,
        experiment=None,
        metrics=Metrics(mean, std),
        global_step=0,
        epoch=0,
        is_main_process=True,
    )


class TestHelpersLiveOnBase(unittest.TestCase):
    """Хелперы подняты в базовый класс — третья копия в SimpleStep не нужна."""

    def test_base_class_exposes_the_physics_hooks(self) -> None:
        for name in (
            "_inner_model",
            "_physics_residual_aux_loss",
            "_physics_residual_diagnostics",
        ):
            self.assertTrue(hasattr(StepStrategy, name), f"StepStrategy лишён {name}")

    def test_aux_loss_is_zero_for_a_model_without_physics(self) -> None:
        strategy = SimpleStep()
        aux = strategy._physics_residual_aux_loss(_PlainModel(), torch.device("cpu"))
        self.assertEqual(aux.item(), 0.0)

    def test_aux_loss_is_read_from_a_pi_model(self) -> None:
        strategy = SimpleStep()
        aux = strategy._physics_residual_aux_loss(_PhysicsModel(), torch.device("cpu"))
        self.assertAlmostEqual(aux.item(), 0.5)

    def test_diagnostics_are_empty_without_physics(self) -> None:
        strategy = SimpleStep()
        self.assertEqual(strategy._physics_residual_diagnostics(_PlainModel()), {})


class TestSimpleStepAddsPhysicsAux(unittest.TestCase):
    """SimpleStep прибавляет физ-штраф к задаче и логирует физ-диагностику."""

    def test_train_step_without_physics_is_unchanged(self) -> None:
        strategy = SimpleStep()
        batch = (torch.ones(2, 4, 69, 8, 16), torch.zeros(2, 4, 69, 8, 16))
        metrics = strategy.train_step(_PlainModel(), batch, _context())
        self.assertEqual(set(metrics), {"loss", "lr"})
        self.assertAlmostEqual(metrics["loss"].item(), 1.0)

    def test_train_step_adds_the_physics_aux_loss(self) -> None:
        strategy = SimpleStep()
        batch = (torch.ones(2, 4, 69, 8, 16), torch.zeros(2, 4, 69, 8, 16))
        metrics = strategy.train_step(_PhysicsModel(), batch, _context())
        # task = MAE(1, 0) = 1.0; aux = 0.5 -> loss = 1.5, и оба видны по отдельности
        self.assertAlmostEqual(metrics["loss"].item(), 1.5)
        self.assertAlmostEqual(metrics["forecast_loss"].item(), 1.0)
        self.assertAlmostEqual(metrics["physics_residual_aux_loss"].item(), 0.5)
        self.assertIn("physics_residual_nonfinite_ratio", metrics)

    def test_physics_aux_loss_reaches_the_head(self) -> None:
        """Регрессия: без этого L1-штраф не доходил бы до оптимизатора."""
        strategy = SimpleStep()
        model = _PhysicsModel()
        batch = (torch.zeros(2, 4, 69, 8, 16), torch.zeros(2, 4, 69, 8, 16))
        metrics = strategy.train_step(model, batch, _context())
        metrics["loss"].backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(model.weight.grad.abs().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
