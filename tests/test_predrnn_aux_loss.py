"""Декаплинг-лосс PredRNNv2 доходит до оптимизатора."""

from __future__ import annotations

import unittest

import torch

from Models.PredRNN import PredRNN_Model, PredRNNv2_Model

CONFIGS = {
    "in_shape": (12, 69, 8, 16),
    "patch_size": 1,
    "filter_size": 3,
    "stride": 1,
    "layer_norm": True,
    "pre_seq_length": 4,
    "aft_seq_length": 4,
    "reverse_scheduled_sampling": 0,
    "decouple_beta": 0.1,
}


def _inputs(total_length: int = 8):
    torch.manual_seed(0)
    frames = torch.randn(2, total_length, 8, 16, 69)
    mask = torch.zeros(1, 4, 1, 1, 1)
    return frames, mask


class TestAuxLossContract(unittest.TestCase):
    def test_v1_returns_empty_aux_dict(self) -> None:
        model = PredRNN_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        self.assertEqual(next_frames.shape, (2, 7, 8, 16, 69))
        self.assertEqual(aux, {})

    def test_v2_returns_weighted_decouple_loss(self) -> None:
        model = PredRNNv2_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        _, aux = model(frames, mask)
        self.assertEqual(set(aux), {"decouple"})
        self.assertEqual(aux["decouple"].ndim, 0)
        self.assertGreater(aux["decouple"].item(), 0.0)

    def test_decouple_beta_scales_the_term(self) -> None:
        frames, mask = _inputs()
        torch.manual_seed(0)
        model_a = PredRNNv2_Model(2, (16, 16), {**CONFIGS, "decouple_beta": 0.1})
        torch.manual_seed(0)
        model_b = PredRNNv2_Model(2, (16, 16), {**CONFIGS, "decouple_beta": 0.2})
        _, aux_a = model_a(frames, mask)
        _, aux_b = model_b(frames, mask)
        torch.testing.assert_close(aux_b["decouple"], 2.0 * aux_a["decouple"])

    def test_adapter_receives_gradient(self) -> None:
        """Регрессия: раньше adapter был мёртв (лосс выбрасывался стратегией)."""
        model = PredRNNv2_Model(2, (16, 16), CONFIGS)
        frames, mask = _inputs()
        next_frames, aux = model(frames, mask)
        loss = next_frames.abs().mean() + aux["decouple"]
        loss.backward()
        self.assertIsNotNone(model.adapter.weight.grad)
        self.assertGreater(model.adapter.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
