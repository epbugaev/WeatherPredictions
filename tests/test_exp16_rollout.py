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
    """model(x, pred_list, t) -> последний кадр x + (len(pred_list)+1).

    Как настоящая IAM4VP, требует contiguous-вход (``x.view(B*T, ...)``).
    """

    def __call__(
        self, x: torch.Tensor, pred_list: list[torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        batch, time = x.shape[0], x.shape[1]
        x.view(batch * time, *x.shape[2:])  # падает на non-contiguous, как модель
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


def test_rollout_handles_noncontiguous_target_slice() -> None:
    # y[:, :horizon] от contiguous y — non-contiguous при B>1; модель требует
    # contiguous (view) → rollout обязан передавать contiguous срез
    horizon = 2
    x = torch.zeros(2, horizon, 1, 2, 2)
    y = torch.arange(2 * 4 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 4, 1, 2, 2)
    free, forced = re_mod.rollout_two_windows(_FakeModel(), x, y, horizon)
    assert free.shape == forced.shape == (2, 4, 1, 2, 2)


def test_predict_teacher_forced_feeds_real_frames() -> None:
    # нативное окно, teacher-forcing: pred_list на шаге i = реальные кадры truth[:i]
    horizon = 3
    x = torch.zeros(1, horizon, 1, 2, 2)
    truth = torch.full((1, horizon, 1, 2, 2), 10.0)
    preds = re_mod.predict_teacher_forced(_FakeModel(), x, truth, horizon)
    assert preds.shape == (1, horizon, 1, 2, 2)
    # шаг 0: pred_list пуст → x[-1]=0 + 1 = 1
    assert torch.allclose(preds[:, 0], torch.full_like(preds[:, 0], 1.0))
    # шаг 1: pred_list=[truth0=10] (len 1) → x[-1]=0 + 2 = 2 (модель видит РЕАЛЬНЫЙ кадр в истории)
    assert torch.allclose(preds[:, 1], torch.full_like(preds[:, 1], 2.0))


def test_rollout_native_free_and_forced_share_first_step() -> None:
    # нативное одно окно: free (свои прогнозы) vs forced (реальные кадры)
    horizon = 3
    x = torch.zeros(1, horizon, 1, 2, 2)
    y = torch.full((1, horizon, 1, 2, 2), 10.0)
    free, forced = re_mod.rollout_native(_FakeModel(), x, y, horizon)
    assert free.shape == forced.shape == (1, horizon, 1, 2, 2)
    # шаг 0 идентичен (pred_list пуст у обоих)
    assert torch.equal(free[:, 0], forced[:, 0])
    # free авторегрессивен (как predict_window) → шаг 2 = 3.0
    assert torch.allclose(free[:, 2], torch.full_like(free[:, 2], 3.0))


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


def test_canonical_arm_both_wave_namings() -> None:
    # t=6 и t=12 имена сводятся к одному ключу
    assert rf_mod.canonical_arm("abl16-r0-no-physics-s0") == "r0-no-physics"
    assert rf_mod.canonical_arm("abl16L-r0-no-physics-t12-s0") == "r0-no-physics"
    assert rf_mod.canonical_arm("abl16L-r3q-diabatic-t-only-t12-s0") == "r3q-diabatic-t-only"
    assert rf_mod.canonical_arm("abl16-r2a-a1-pre13-s0") == "r2a-a1-pre13"


def test_window_boundary_two_window_vs_native() -> None:
    # t=6: native_horizon 6 < 12 шагов → граница 6; t=12: 12 == 12 → None
    two_window = {"r0-no-physics": {"rmse_free": np.zeros((12, 3)), "native_horizon": 6}}
    native = {"r0-no-physics": {"rmse_free": np.zeros((12, 3)), "native_horizon": 12}}
    assert rf_mod.window_boundary(two_window) == 6
    assert rf_mod.window_boundary(native) is None


def test_epoch_label_fixed_epoch_wave() -> None:
    # волна на общей эпохе: все армы с одного тега → одна подпись (тег 89 = эпоха 90)
    runs = {"r0-no-physics": {"checkpoint_epoch": 89}, "r4-exp14": {"checkpoint_epoch": 89}}
    assert rf_mod.epoch_label(runs) == "эпоха 90"


def test_epoch_label_best_val_selection_reports_range() -> None:
    # отбор по валидации: у армов РАЗНЫЕ эпохи → подпись обязана показать диапазон,
    # а не эпоху одного baseline (иначе фигура врёт про остальные 8 армов)
    runs = {"r0-no-physics": {"checkpoint_epoch": 269}, "r4-exp14": {"checkpoint_epoch": 208}}
    assert rf_mod.epoch_label(runs) == "лучший val, эпохи 209–270"


def test_epoch_label_without_provenance() -> None:
    runs = {"r0-no-physics": {"checkpoint_epoch": -1}}
    assert rf_mod.epoch_label(runs) == "эпоха ?"


def test_level_step_delta_matrix_against_baseline() -> None:
    # Δ% к R0 на сетке [уровень × шаг]: строки — уровни по возрастанию, столбцы — шаги
    channels = ["z50", "z500"]
    runs = {
        "r0-no-physics": {
            "rmse_free": np.array([[10.0, 20.0], [10.0, 20.0]]),
            "channels": channels,
        },
        "r3-a2-exp13": {"rmse_free": np.array([[9.0, 20.0], [11.0, 20.0]]), "channels": channels},
    }
    delta, levels = rf_mod.level_step_delta(runs, "r3-a2-exp13", "z")
    assert levels == [50, 500]
    assert delta.shape == (2, 2)
    assert delta[0].tolist() == [-10.0, 10.0]  # z50: шаг 1 лучше R0, шаг 2 хуже
    assert delta[1].tolist() == [0.0, 0.0]  # z500 не отличается от R0
