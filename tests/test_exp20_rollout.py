"""Пины rollout-механики exp20 (PI-PredRNNv2): маска, срез окна, агрегация RMSE.

Три места, где ошибка молчалива (метрики считаются, npz пишется, цифры врут):

1. Семантика ``mask_true``: нули => после контекста модель кормит себя своими
   прогнозами (free-running), единицы => на каждом шаге входит реальный кадр
   (teacher forcing). Перепутанные режимы дают правдоподобные, но неверные
   кривые.
2. Срез окна прогноза: PredRNN отдаёт ``T-1`` кадров, кадр ``i`` — прогноз
   кадра ``i+1``, поэтому прогноз начинается с индекса ``x.shape[1] - 1``.
   Сдвиг на единицу переставил бы все лид-таймы.
3. Форма и физические единицы агрегированного lat-weighted RMSE.

Инференс-цикл (чекпоинты, memmap) не тестируется — только эта механика.
"""

from __future__ import annotations

import unittest

import torch

from Models.PredRNN import PI_PredRNNv2_Model
from tools.rollout_common import LatWeightedRmseAccumulator, channel_names
from training_strategies.predrnn import predrnn_forecast

PRE_SEQ, AFT_SEQ = 2, 2
HEIGHT, WIDTH, CHANNELS = 32, 64, 69

TOY_CONFIGS = {
    "in_shape": (PRE_SEQ, CHANNELS, HEIGHT, WIDTH),
    "patch_size": 1,
    "filter_size": 3,
    "stride": 1,
    "layer_norm": True,
    "pre_seq_length": PRE_SEQ,
    "aft_seq_length": AFT_SEQ,
    "reverse_scheduled_sampling": 0,
    "decouple_beta": 0.1,
}

TOY_PHYSICS = {
    "use_physics_residual_corrector": True,
    "physics_residual_hidden_channels": 16,
    "physics_residual_apply_to": "upper_air_only",
    "physics_residual_zero_init": False,
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


class _SpyPI(PI_PredRNNv2_Model):
    """PI-PredRNNv2, запоминающий вход КАЖДОГО шага авторегрессии.

    ``net`` в :meth:`_correct_step` — это ровно то, что модель получила на вход
    на данном шаге: либо реальный кадр, либо собственный прогноз (в зависимости
    от маски). Иначе режим rollout снаружи не наблюдаем.
    """

    def __init__(self, num_layers: int, num_hidden, configs: dict, **physics_kwargs) -> None:
        super().__init__(num_layers, num_hidden, configs, **physics_kwargs)
        self.step_inputs: list[torch.Tensor] = []

    def _correct_step(self, x_gen: torch.Tensor, net: torch.Tensor) -> torch.Tensor:
        self.step_inputs.append(net.detach().clone())
        return super()._correct_step(x_gen, net)


class _OracleModel:
    """Идеальный прогнозист с контрактом PredRNN: ``next_frames[i] = frames[i+1]``.

    Позволяет отделить срез окна от качества модели: правильный срез обязан
    вернуть РОВНО таргет ``y``, любой off-by-one — сдвинутую копию.
    """

    def __call__(
        self, frames: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return frames[:, 1:], {}


def _build_spy() -> _SpyPI:
    torch.manual_seed(0)
    model = _SpyPI(2, (16, 16), TOY_CONFIGS, **TOY_PHYSICS)
    model.eval()
    model.set_physics_normalization(torch.zeros(CHANNELS), torch.ones(CHANNELS))
    return model


def _clips() -> tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)``: контекст и таргет в channels-first ``(B, T, C, H, W)``."""
    torch.manual_seed(1)
    x = torch.randn(1, PRE_SEQ, CHANNELS, HEIGHT, WIDTH)
    y = torch.randn(1, AFT_SEQ, CHANNELS, HEIGHT, WIDTH)
    return x, y


class TestForecastWindowSlicing(unittest.TestCase):
    def test_step_one_is_the_first_target_frame(self) -> None:
        """Прогноз стартует с кадра, предсказанного из ПОСЛЕДНЕГО кадра контекста."""
        x, y = _clips()
        forecast = predrnn_forecast(_OracleModel(), x, y)
        self.assertEqual(forecast.shape, y.shape)
        # Оракул предсказывает следующий кадр точно => правильный срез == y.
        # Сдвиг на -1 вернул бы x[:, -1] первым кадром, на +1 — потерял бы y[:, 0].
        torch.testing.assert_close(forecast, y)
        self.assertFalse(torch.equal(forecast[:, 0], x[:, -1]))

    def test_shape_on_the_real_model(self) -> None:
        model = _build_spy()
        x, y = _clips()
        with torch.inference_mode():
            forecast = predrnn_forecast(model, x, y)
        self.assertEqual(forecast.shape, (1, AFT_SEQ, CHANNELS, HEIGHT, WIDTH))


class TestMaskSemantics(unittest.TestCase):
    def test_free_running_feeds_the_models_own_prediction(self) -> None:
        """mask=0: вход первого шага ПОСЛЕ контекста == собственный прогноз шага 1."""
        model = _build_spy()
        x, y = _clips()
        with torch.inference_mode():
            free = predrnn_forecast(model, x, y, teacher_forcing=False)
        # step_inputs[PRE_SEQ] — первый шаг, где маска вообще действует.
        torch.testing.assert_close(model.step_inputs[PRE_SEQ], free[:, 0])
        self.assertFalse(torch.allclose(model.step_inputs[PRE_SEQ], y[:, 0]))

    def test_teacher_forcing_feeds_the_real_frame(self) -> None:
        """mask=1: вход того же шага == реальный кадр ``y[:, 0]``."""
        model = _build_spy()
        x, y = _clips()
        with torch.inference_mode():
            forced = predrnn_forecast(model, x, y, teacher_forcing=True)
        torch.testing.assert_close(model.step_inputs[PRE_SEQ], y[:, 0])
        self.assertFalse(torch.allclose(model.step_inputs[PRE_SEQ], forced[:, 0]))

    def test_two_modes_differ_beyond_the_first_step(self) -> None:
        """Шаг 1 общий (обе истории — реальный контекст), дальше режимы расходятся."""
        model = _build_spy()
        x, y = _clips()
        with torch.inference_mode():
            free = predrnn_forecast(model, x, y, teacher_forcing=False)
            forced = predrnn_forecast(model, x, y, teacher_forcing=True)
        torch.testing.assert_close(free[:, 0], forced[:, 0])
        self.assertGreater((free[:, 1] - forced[:, 1]).abs().max().item(), 0.0)

    def test_free_running_never_reads_the_target_clip(self) -> None:
        """Регрессия: при маске 0 таргет — только контейнер длины горизонта.

        Если маска однажды перестанет быть нулевой, free-running втихую начнёт
        подглядывать в будущее, и скилл окажется завышен.
        """
        model = _build_spy()
        x, y = _clips()
        with torch.inference_mode():
            free = predrnn_forecast(model, x, y, teacher_forcing=False)
            free_other_target = predrnn_forecast(model, x, torch.randn_like(y))
            forced = predrnn_forecast(model, x, y, teacher_forcing=True)
            forced_other_target = predrnn_forecast(model, x, torch.randn_like(y), True)
        torch.testing.assert_close(free, free_other_target)
        self.assertGreater((forced - forced_other_target).abs().max().item(), 0.0)


class TestRmseAggregation(unittest.TestCase):
    def test_shape_is_steps_by_channels(self) -> None:
        std = torch.ones(4)
        accumulator = LatWeightedRmseAccumulator(
            rollout_steps=12, std=std, device=torch.device("cpu")
        )
        pred = torch.zeros(3, 12, 4, 8, 16)
        accumulator.update(pred, torch.zeros_like(pred))
        self.assertEqual(accumulator.mean().shape, (12, 4))
        self.assertEqual(accumulator.n_samples, 3)

    def test_rmse_is_denormalized_by_std(self) -> None:
        """RMSE в нормированном пространстве × std == RMSE в физических единицах."""
        std = torch.tensor([2.0, 10.0])
        accumulator = LatWeightedRmseAccumulator(
            rollout_steps=2, std=std, device=torch.device("cpu")
        )
        target = torch.zeros(1, 2, 2, 8, 16)
        accumulator.update(torch.ones_like(target), target)  # ошибка 1.0 в норм. единицах
        mean = accumulator.mean()
        self.assertAlmostEqual(float(mean[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(mean[1, 1]), 10.0, places=5)

    def test_mean_accumulates_over_batches(self) -> None:
        std = torch.ones(1)
        accumulator = LatWeightedRmseAccumulator(
            rollout_steps=1, std=std, device=torch.device("cpu")
        )
        target = torch.zeros(2, 1, 1, 4, 8)
        accumulator.update(torch.full_like(target, 1.0), target)
        accumulator.update(torch.full_like(target[:1], 3.0), target[:1])
        # средневзвешенное по сэмплам: (2*1 + 1*3) / 3
        self.assertAlmostEqual(float(accumulator.mean()[0, 0]), 5.0 / 3.0, places=5)
        self.assertEqual(accumulator.n_samples, 3)


class TestChannelNames(unittest.TestCase):
    def test_v4_layout(self) -> None:
        names = channel_names()
        self.assertEqual(len(names), CHANNELS)
        self.assertEqual(names[:4], ["t2", "u10", "v10", "tp"])
        self.assertEqual(names[4], "z50")


if __name__ == "__main__":
    unittest.main()
