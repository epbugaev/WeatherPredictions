"""Пины чистой логики exp16 rollout: окна прогноза, имена каналов, дельты фигур.

Инференс-цикл (GPU, чекпоинты) не тестируется — только оконная механика и
преобразования метрик, где живут баги. Модули грузятся по пути:
``docs/experiments`` не является пакетом.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "docs/experiments/16_model_ablation_ladder"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


re_mod = _load("exp16_rollout_eval", "rollout_eval.py")
rf_mod = _load("exp16_rollout_figures", "rollout_figures.py")


class _FakeModel:
    """model(x, pred_list, t) -> последний кадр x + (len(pred_list)+1)."""

    def __call__(
        self, x: torch.Tensor, pred_list: list[torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        return x[:, -1] + float(len(pred_list) + 1)


def test_predict_window_shapes_and_autoregression() -> None:
    x = torch.zeros(2, 3, 4, 5, 6)
    preds = re_mod.predict_window(_FakeModel(), x, horizon=3)
    assert preds.shape == (2, 3, 4, 5, 6)
    # фейк-модель добавляет (len(pred_list)+1) к последнему кадру x=0
    assert torch.allclose(preds[:, 0], torch.full_like(preds[:, 0], 1.0))
    assert torch.allclose(preds[:, 2], torch.full_like(preds[:, 2], 3.0))


def test_rollout_two_windows_boundary_semantics() -> None:
    horizon = 2
    x = torch.zeros(1, horizon, 1, 2, 2)
    y = torch.full((1, 2 * horizon, 1, 2, 2), 10.0)
    free, forced = re_mod.rollout_two_windows(_FakeModel(), x, y, horizon)
    assert free.shape == (1, 4, 1, 2, 2)
    assert forced.shape == (1, 4, 1, 2, 2)
    # первое окно общее
    assert torch.equal(free[:, :horizon], forced[:, :horizon])
    # окно 2 free стартует с СОБСТВЕННЫХ прогнозов (последний = 2.0) -> 2+1=3
    assert torch.allclose(free[:, horizon], torch.full_like(free[:, horizon], 3.0))
    # окно 2 forced стартует с реальных кадров y (=10) -> 10+1=11
    assert torch.allclose(forced[:, horizon], torch.full_like(forced[:, horizon], 11.0))


def test_channel_names_layout() -> None:
    names = re_mod.channel_names()
    assert len(names) == 69
    assert names[:4] == ["t2", "u10", "v10", "tp"]
    assert names[4] == "z50"
    assert names[16] == "z1000"
    assert names[30] == "r50"
    assert names[68] == "v1000"


def test_delta_percent_sign() -> None:
    base = np.full((2, 3), 10.0)
    arm = np.array([[9.0, 10.0, 11.0], [10.0, 10.0, 10.0]])
    delta = rf_mod.delta_percent(arm, base)
    assert delta.shape == (2, 3)
    assert delta[0, 0] == -10.0
    assert delta[0, 2] == 10.0


def test_level_matrix_orders_levels_ascending() -> None:
    channels = ["t2", "z1000", "z50", "z500"]
    rmse = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )  # (steps=2, C=4)
    matrix, levels = rf_mod.level_matrix(rmse, channels, "z")
    assert levels == [50, 500, 1000]
    # строки = уровни по возрастанию, столбцы = шаги
    assert matrix.shape == (3, 2)
    assert matrix[0].tolist() == [3.0, 7.0]  # z50
    assert matrix[2].tolist() == [2.0, 6.0]  # z1000


def test_mean_delta_over_channels() -> None:
    delta = np.array([[1.0, 3.0], [-2.0, 2.0]])  # (steps, C)
    mean = rf_mod.mean_delta_over_channels(delta)
    assert mean.tolist() == [2.0, 0.0]
