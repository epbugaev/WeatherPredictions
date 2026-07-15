"""Архитектурно-нейтральное ядро rollout-инференса арма из чекпоинта трейнера.

Одинаково для лестниц exp16 (PI-IAM4VP) и exp20 (PI-PredRNNv2): имена каналов
v4-раскладки, вал-датасет с таргетом на весь горизонт rollout, восстановление
модели и нормализации из payload трейнера, аккумуляция lat-weighted RMSE в
физических единицах и схема выходного npz. Специфика форварда — чем модель
авторегрессирует и что означает teacher-forcing — остаётся в скрипте
эксперимента: у IAM4VP это ``model(x, pred_list, t)``, у PredRNN — маска
scheduled sampling внутри одного форварда.

Схема npz (её потребляют ``rollout_figures.py`` обоих экспериментов):
``rmse_free``/``rmse_forced`` формы ``(steps, C)`` в физических единицах,
``channels``, ``n_samples``, ``checkpoint_epoch``, ``native_horizon``, ``arm``.

Скрипт exp16 (``docs/experiments/16_model_ablation_ladder/rollout_eval.py``)
этот модуль намеренно НЕ импортирует и держит свою копию помощников: его армы
R2a/R2 инферируются кодом pre-exp13 worktree (``REPO_ROOT=${PREFIX13_ROOT}``,
см. ``sh_files/abl16_rollout_eval.sh``), где данного файла нет — импорт сломал
бы ровно эти два арма.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import Data  # noqa: F401  (импорт регистрирует датасеты в utils.registry)
import Models  # noqa: F401  (импорт регистрирует модели в utils.registry)
from training_strategies._index_maps import MULTIOUT_INDEX_MAP
from utils.metrics import weighted_rmse_torch_channels
from utils.normalize import WeatherNormalize
from utils.registry import get_dataset, get_model


@dataclass(frozen=True)
class ArmCheckpoint:
    """Восстановленный из чекпоинта арм: модель, нормализация, конфиг, происхождение.

    Attributes:
        model: модель в ``eval``-режиме на целевом device.
        normalize: ``WeatherNormalize`` арма на том же device.
        config: конфиг рана (блоки ``model``/``data``/``training``).
        arm: имя арма (``experiment.name``; иначе имя каталога рана).
        epoch: эпоха чекпоинта (``-1``, если payload её не несёт).
    """

    model: nn.Module
    normalize: WeatherNormalize
    config: dict[str, Any]
    arm: str
    epoch: int


def channel_names() -> list[str]:
    """69 имён каналов в порядке индексов выходного тензора (v4-layout)."""
    by_index = {index: name for name, index in MULTIOUT_INDEX_MAP.items()}
    return [by_index[i] for i in range(len(by_index))]


def load_arm_checkpoint(checkpoint_path: str, device: torch.device) -> ArmCheckpoint:
    """Собрать модель и нормализацию арма из чекпоинта трейнера.

    Модель строится по ``config`` из самого payload (а не по конфигу с диска),
    поэтому скрипт не зависит от того, менялись ли конфиги после обучения.
    Модели с физветкой получают статистики нормализации через
    ``set_physics_normalization`` — без этого физядро считает по нормированному
    состоянию как по физическому.

    Args:
        checkpoint_path: путь к ``.pt`` (``last.pt``/``best.pt`` трейнера).
        device: куда положить модель и нормализацию.

    Returns:
        :class:`ArmCheckpoint`.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    arm = config.get("experiment", {}).get("name", Path(checkpoint_path).parent.name)

    model = get_model(config["model"]["type"])(**config["model"].get("params", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    n_channels = len(channel_names())
    normalize = WeatherNormalize(mean=torch.zeros(n_channels), std=torch.ones(n_channels))
    normalize.load_state_dict(payload["normalize"])
    normalize.to(device)

    set_physics_normalization = getattr(model, "set_physics_normalization", None)
    if set_physics_normalization is not None:
        set_physics_normalization(normalize.mean.view(-1).cpu(), normalize.std.view(-1).cpu())

    return ArmCheckpoint(
        model=model,
        normalize=normalize,
        config=config,
        arm=arm,
        epoch=int(payload.get("epoch", -1)),
    )


def build_val_dataset(data_cfg: dict[str, Any], memmap_path: str, rollout_steps: int) -> Dataset:
    """Вал-датасет арма с таргетом длиной ``rollout_steps`` кадров.

    Повторяет ``train.build_dataset(data_cfg, "val")`` для v4-memmap, меняя
    только ``end_time_y`` (под нужный горизонт) и путь memmap.

    Args:
        data_cfg: блок ``data`` конфига из чекпоинта.
        memmap_path: путь к packed-memmap ``.dat`` (meta — рядом,
            ``<stem>.meta.json``).
        rollout_steps: полное число шагов rollout.

    Returns:
        Датасет: сэмпл ``(x (T_in, C, H, W), y (rollout_steps, C, H, W))``,
        raw float32 (ненормированный).
    """
    split_cfg = data_cfg.get("val", {})
    start_time_y = int(data_cfg.get("start_time_y", 6))
    params: dict[str, Any] = {
        "start_time": split_cfg["start_time"],
        "end_time": split_cfg["end_time"],
        "include_target": split_cfg.get("include_target", data_cfg.get("include_target", False)),
        "lead_time": split_cfg.get("lead_time", data_cfg.get("lead_time", 1)),
        "interval": split_cfg.get("interval", data_cfg.get("interval", 1)),
        "muti_target_steps": split_cfg.get(
            "muti_target_steps", data_cfg.get("muti_target_steps", 1)
        ),
        "start_time_x": data_cfg.get("start_time_x", 0),
        "end_time_x": data_cfg.get("end_time_x", 5),
        "start_time_y": start_time_y,
        "end_time_y": start_time_y + rollout_steps - 1,
        "memmap_path": memmap_path,
        "memmap_meta_path": str(Path(memmap_path).with_suffix("")) + ".meta.json",
    }
    for key in ("sample_stride", "frame_interval"):
        value = split_cfg.get(key, data_cfg.get(key))
        if value is not None:
            params[key] = value
    cut = split_cfg.get("cut", data_cfg.get("cut"))
    if cut is not None:
        params["cut"] = cut
    return get_dataset(data_cfg.get("dataset_version", "v4"))(**params)


class LatWeightedRmseAccumulator:
    """Сумма lat-weighted RMSE по (шаг, канал) → среднее по сэмплам в физединицах.

    RMSE считается в нормированном пространстве и домножается на ``std`` канала:
    для поканальной нормировки это и есть RMSE в физических единицах (сдвиг
    ``mean`` из разности выпадает).

    Args:
        rollout_steps: число шагов прогноза (строк в матрице).
        std: ``(C,)`` — стандартные отклонения каналов нормировки арма.
        device: device аккумулятора (совпадает с device модели).
    """

    def __init__(self, rollout_steps: int, std: torch.Tensor, device: torch.device) -> None:
        self._std = std.reshape(-1).to(device)
        self._sum = torch.zeros(rollout_steps, self._std.numel(), device=device)
        self.n_samples = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Добавить батч.

        Args:
            pred: прогноз ``(B, steps, C, H, W)`` в нормированном пространстве.
            target: эталон той же формы.

        Side effects: мутирует накопленную сумму и ``n_samples``.
        """
        self._sum += (weighted_rmse_torch_channels(pred, target) * self._std).sum(dim=0)
        self.n_samples += pred.shape[0]

    def mean(self) -> np.ndarray:
        """Средний по валидации RMSE, ``(steps, C)`` в физических единицах."""
        assert self.n_samples > 0, "LatWeightedRmseAccumulator.mean() до единого батча"
        return (self._sum / self.n_samples).cpu().numpy()


def save_rollout_npz(
    out_path: str,
    rmse_free: np.ndarray,
    rmse_forced: np.ndarray,
    n_samples: int,
    checkpoint_epoch: int,
    native_horizon: int,
    arm: str,
) -> None:
    """Записать npz общей схемы (см. docstring модуля).

    Args:
        out_path: путь к ``.npz``; родительские каталоги создаются.
        rmse_free: ``(steps, C)`` — free-running RMSE, физические единицы.
        rmse_forced: ``(steps, C)`` — teacher-forced RMSE, физические единицы.
        n_samples: число вал-сэмплов, по которым усреднено.
        checkpoint_epoch: эпоха чекпоинта (для подписи фигур).
        native_horizon: нативный горизонт модели (граница окон у фигур).
        arm: имя арма.

    Side effects: пишет файл на диск.
    """
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        rmse_free=rmse_free,
        rmse_forced=rmse_forced,
        channels=np.array(channel_names()),
        n_samples=n_samples,
        checkpoint_epoch=checkpoint_epoch,
        native_horizon=native_horizon,
        arm=arm,
    )
