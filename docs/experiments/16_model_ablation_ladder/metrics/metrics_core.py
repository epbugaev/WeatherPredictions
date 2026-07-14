"""Архитектурно-нейтральное ядро полного набора метрик арма лестницы.

Общее для exp16 (PI-IAM4VP) и exp20 (PI-PredRNNv2): восстановление арма из чекпоинта,
вал-датасет, календарь таргет-кадров (для климатологии ACC), сам прогон метрик по
(сэмпл × шаг × канал), бутстрап-сводка и **схема выходного npz**. Специфика остаётся
в скрипте эксперимента и сводится к одной функции ``forecast(x_norm, y_norm)``:
у IAM4VP это внешняя авторегрессия ``model(x, pred_list, t)``, у PredRNN — маска
scheduled sampling внутри одного форварда.

Считается по каждому (сэмпл × шаг × канал), чтобы бутстрап-CI строился снаружи:

  * ``rmse``, ``acc``, ``bias``, ``w1`` — по-сэмпльно ``(S, T, C)``;
  * ``std_pred`` / ``std_obs`` — пространственный std поля (проверка на сглаживание);
  * ``csi_tp/fp/fn`` — взвешенные счётчики на пороги p90/p95/p99 ``(S, T, C, 3)``
    (агрегировать надо счётчики, а не готовый CSI: сэмплы без событий его ломают);
  * ``fss_num/den`` на те же пороги и окна 1/3/5 ячеек ``(S, T, C, 3, 3)``;
  * ``psd_pred`` / ``psd_obs`` — спектр по зональному волновому числу m ``(T, C, W//2+1)``.

Почему модуль лежит здесь, а не в ``tools/`` (как ``tools/rollout_common.py``): армы
R2a/R2 эксперимента 16 инферируются кодом pre-exp13 worktree (``REPO_ROOT=${PREFIX13_ROOT}``,
см. ``sh_files/abl16_rollout_eval.sh``), где каталога ``tools/`` в нынешнем виде нет.
Скрипты грузят этот модуль **по пути к файлу** (importlib), поэтому он доступен при
любом ``sys.path``; ценой этого он не может импортировать ``tools.rollout_common`` и
держит собственные копии загрузчика чекпоинта и вал-датасета.
"""

from __future__ import annotations

import importlib.util
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import Data  # noqa: F401  (импорт регистрирует датасеты в utils.registry)
import Models  # noqa: F401  (импорт регистрирует модели в utils.registry)
from training_strategies._index_maps import MULTIOUT_INDEX_MAP
from utils.normalize import WeatherNormalize
from utils.registry import get_dataset, get_model

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16_metrics_lib", HERE / "metrics_lib.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

QUANTILES = (0.90, 0.95, 0.99)  # пороги событий CSI/FSS (по истине, общие для армов)
NEIGHBORHOODS = (1, 3, 5)  # окна FSS в ячейках (1 = по-ячеечно)

# Прогноз арма: нормированный контекст и нормированный таргет → нормированный прогноз
# ``(B, T, C, H, W)``. Таргет нужен не всем: IAM4VP берёт из него только длину окна,
# PredRNN — ещё и кадры (при teacher forcing; при free-running маска их обнуляет).
ForecastFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ArmCheckpoint:
    """Восстановленный из чекпоинта арм.

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
    by_index = {idx: name for name, idx in MULTIOUT_INDEX_MAP.items()}
    return [by_index[i] for i in range(len(by_index))]


def load_arm(checkpoint_path: str, device: torch.device) -> ArmCheckpoint:
    """Собрать модель и нормализацию арма из чекпоинта трейнера.

    Модель строится по ``config`` из самого payload (а не по конфигу с диска), поэтому
    скрипт не зависит от того, менялись ли конфиги после обучения. Модели с физветкой
    получают статистики нормализации через ``set_physics_normalization`` — без этого
    физядро считало бы по нормированному состоянию как по физическому.

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
    """Вал-датасет арма с таргетом, удлинённым до ``rollout_steps`` кадров.

    Повторяет ``train.build_dataset(data_cfg, "val")`` для v4-memmap, меняя только
    ``end_time_y`` и путь. Датасет нужен ещё и как источник времени: климатология для
    ACC ищется по дню года каждого таргет-кадра.

    Args:
        data_cfg: блок ``data`` конфига из чекпоинта.
        memmap_path: путь к packed-memmap ``.dat`` (meta — рядом, ``<stem>.meta.json``).
        rollout_steps: полное число шагов rollout.

    Returns:
        Датасет: сэмпл ``(x (T_in, C, H, W), y (rollout_steps, C, H, W))``, raw float32.
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


def target_day_of_year(dataset: Dataset, rollout_steps: int, start_time_y: int) -> np.ndarray:
    """День года (0..365) каждого таргет-кадра: ``(S, rollout_steps)``.

    Климатология лежит по дню года, поэтому для аномалий нужно знать, на какую дату
    приходится каждый шаг прогноза каждого сэмпла.
    """
    calendar = dataset.x_time_ilst
    starts = dataset.sample_start_indices
    offsets = np.arange(rollout_steps) + start_time_y
    days = np.empty((len(starts), rollout_steps), dtype=np.int64)
    for row, start_idx in enumerate(starts):
        stamps = pd.DatetimeIndex([calendar[start_idx + int(off)] for off in offsets])
        days[row] = stamps.dayofyear.to_numpy() - 1
    return days


def evaluate_arm(
    arm_checkpoint: ArmCheckpoint,
    forecast: ForecastFn,
    rollout_steps: int,
    native_horizon: int,
    args: Namespace,
) -> None:
    """Прогнать валидацию арма, посчитать все метрики и записать npz.

    Args:
        arm_checkpoint: модель, нормализация и конфиг арма (см. :func:`load_arm`).
        forecast: ``(x_norm, y_norm) -> pred_norm`` формы ``(B, rollout_steps, C, H, W)`` —
            единственное, чем архитектуры отличаются.
        rollout_steps: длина окна прогноза (число шагов диагностики).
        native_horizon: нативный горизонт модели (пишется в npz, подписывает фигуры).
        args: разобранный CLI (см. :func:`parse_args`).

    Side effects: пишет ``args.out`` (сводка с CI) и, если задан, ``args.out_per_sample``;
    печатает прогресс и sanity-строку.
    """
    device = next(arm_checkpoint.model.parameters()).device
    arm, normalize = arm_checkpoint.arm, arm_checkpoint.normalize
    start_time_y = int(arm_checkpoint.config["data"].get("start_time_y", 6))

    dataset = build_val_dataset(arm_checkpoint.config["data"], args.memmap, rollout_steps)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    days = target_day_of_year(dataset, rollout_steps, start_time_y)

    clim_npz = np.load(args.climatology)
    climatology = torch.as_tensor(clim_npz["climatology"], device=device)  # (366, C, H, W)

    names = channel_names()
    n_channels = len(names)
    weights = ml.lat_weights(climatology.shape[-2], device=device)
    thresholds = torch.as_tensor(np.load(args.thresholds)["thresholds"], device=device)  # (C, K)

    per_sample: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("rmse", "acc", "bias", "w1", "std_pred", "std_obs")
    }
    counts: dict[str, list[torch.Tensor]] = {key: [] for key in ("tp", "fp", "fn")}
    fss_num: list[torch.Tensor] = []
    fss_den: list[torch.Tensor] = []
    psd_pred_sum = torch.zeros(rollout_steps, n_channels, climatology.shape[-1] // 2 + 1)
    psd_obs_sum = torch.zeros_like(psd_pred_sum)
    n_samples = 0

    with torch.inference_mode():
        for batch_idx, (x_raw, y_raw) in enumerate(loader):
            batch = x_raw.shape[0]
            rows = slice(n_samples, n_samples + batch)
            x = normalize(x_raw.to(device, non_blocking=True))
            y_physical = y_raw.to(device, non_blocking=True)
            obs = y_physical[:, :rollout_steps]  # физ. единицы
            pred_norm = forecast(x, normalize(y_physical)[:, :rollout_steps])
            pred = normalize.denormalize(pred_norm)  # обратно в физические единицы

            batch_days = torch.as_tensor(days[rows], device=device)  # (B, T)
            clim = climatology[batch_days]  # (B, T, C, H, W)
            pred_anom, obs_anom = pred - clim, obs - clim

            per_sample["rmse"].append(ml.weighted_rmse(pred, obs, weights).cpu())
            per_sample["acc"].append(ml.weighted_acc(pred_anom, obs_anom, weights).cpu())
            per_sample["bias"].append(ml.weighted_bias(pred, obs, weights).cpu())
            per_sample["w1"].append(ml.wasserstein1(pred, obs).cpu())
            per_sample["std_pred"].append(ml.weighted_std(pred, weights).cpu())
            per_sample["std_obs"].append(ml.weighted_std(obs, weights).cpu())

            tp, fp, fn = ml.contingency_counts(pred, obs, thresholds, weights)
            counts["tp"].append(tp.cpu())
            counts["fp"].append(fp.cpu())
            counts["fn"].append(fn.cpu())

            num_stack, den_stack = [], []
            for neighborhood in NEIGHBORHOODS:
                num, den = ml.fss_terms(pred, obs, thresholds, neighborhood)
                num_stack.append(num)
                den_stack.append(den)
            fss_num.append(torch.stack(num_stack, dim=-1).cpu())  # (B,T,C,K,N)
            fss_den.append(torch.stack(den_stack, dim=-1).cpu())

            psd_pred_sum += ml.zonal_psd(pred, weights).sum(dim=0).cpu()
            psd_obs_sum += ml.zonal_psd(obs, weights).sum(dim=0).cpu()

            n_samples += batch
            if batch_idx % 10 == 0:
                print(f"[metrics] {arm}: batch {batch_idx}, samples {n_samples}")  # noqa: T201

    raw = {key: torch.cat(values).numpy() for key, values in per_sample.items()}
    tp = torch.cat(counts["tp"]).numpy()
    fp = torch.cat(counts["fp"]).numpy()
    fn = torch.cat(counts["fn"]).numpy()
    num = torch.cat(fss_num).numpy()
    den = torch.cat(fss_den).numpy()

    z500, t850 = names.index("z500"), names.index("t850")
    print(  # noqa: T201
        f"[sanity] {arm}: RMSE z500 шаг1={raw['rmse'][:, 0, z500].mean():.2f} "
        f"ACC z500 шаг1={raw['acc'][:, 0, z500].mean():.3f} "
        f"ACC t850 шаг{rollout_steps}={raw['acc'][:, -1, t850].mean():.3f}"
    )

    summary = bootstrap_summary(raw, tp, fp, fn, num, den, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **summary,
        psd_pred=(psd_pred_sum / n_samples).numpy(),
        psd_obs=(psd_obs_sum / n_samples).numpy(),
        thresholds=thresholds.cpu().numpy(),
        quantiles=np.array(QUANTILES),
        neighborhoods=np.array(NEIGHBORHOODS),
        channels=np.array(names),
        latitudes_deg=ml.crop_latitudes_deg(climatology.shape[-2]),
        n_samples=n_samples,
        n_bootstrap=args.bootstrap,
        checkpoint_epoch=arm_checkpoint.epoch,
        native_horizon=native_horizon,
        arm=arm,
    )
    print(f"[metrics] {arm}: сводка → {out_path} (n={n_samples})")  # noqa: T201

    if args.out_per_sample:
        per_sample_path = Path(args.out_per_sample)
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            per_sample_path, **raw, csi_tp=tp, csi_fp=fp, csi_fn=fn, fss_num=num, fss_den=den
        )
        print(f"[metrics] {arm}: по-сэмпльно → {per_sample_path}")  # noqa: T201


def bootstrap_summary(
    raw: dict[str, np.ndarray],
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    fss_num: np.ndarray,
    fss_den: np.ndarray,
    args: Namespace,
) -> dict[str, np.ndarray]:
    """Бутстрап-сводка арма: среднее, std и 95% CI каждой метрики.

    Бутстрап делается здесь, а не при анализе: по-сэмпльные массивы весят десятки МБ
    на арм и в репозиторий не поедут, а сводка — сотни КБ.

    Args:
        raw: по-сэмпльные метрики ``(S, T, C)`` (ключи ``rmse``/``acc``/``bias``/``w1``/
            ``std_pred``/``std_obs``).
        tp: взвешенные попадания CSI ``(S, T, C, K)``.
        fp: ложные тревоги CSI той же формы.
        fn: пропуски CSI той же формы.
        fss_num: числитель FSS ``(S, T, C, K, N)``.
        fss_den: знаменатель FSS той же формы.
        args: CLI (нужны ``bootstrap`` и ``seed``).

    Returns:
        Плоский словарь полей npz (``<метрика>_mean/_std/_ci_low/_ci_high`` и агрегаты).
    """
    summary: dict[str, np.ndarray] = {}
    for key in ("rmse", "acc", "bias", "w1", "std_pred", "std_obs"):
        mean, std, low, high = ml.bootstrap_ci(raw[key], args.bootstrap, seed=args.seed)
        summary |= {
            f"{key}_mean": mean,
            f"{key}_std": std,
            f"{key}_ci_low": low,
            f"{key}_ci_high": high,
        }
    csi_mean, csi_std, csi_low, csi_high = ml.bootstrap_ratio_ci(
        tp, tp + fp + fn, args.bootstrap, seed=args.seed
    )
    fss_ratio = ml.bootstrap_ratio_ci(fss_num, fss_den, args.bootstrap, seed=args.seed)
    return summary | {
        "csi_mean": csi_mean,
        "csi_std": csi_std,
        "csi_ci_low": csi_low,
        "csi_ci_high": csi_high,
        "mcsi_mean": np.nanmean(csi_mean, axis=-1),  # среднее по порогам p90/p95/p99
        "fss_mean": 1.0 - fss_ratio[0],
        "fss_std": fss_ratio[1],
        "fss_ci_low": 1.0 - fss_ratio[3],  # 1−x переворачивает границы интервала
        "fss_ci_high": 1.0 - fss_ratio[2],
    }


def parse_args(description: str) -> Namespace:
    """CLI харнесса: чекпоинт, memmap, климатология, пороги, выходные npz.

    Args:
        description: docstring скрипта эксперимента (идёт в ``--help``).

    Returns:
        Разобранные аргументы.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True, help="путь к .pt чекпоинту")
    parser.add_argument("--memmap", required=True, help="packed memmap .dat")
    parser.add_argument("--climatology", required=True, help="npz климатологии (climatology.py)")
    parser.add_argument("--thresholds", required=True, help="npz порогов (thresholds.py)")
    parser.add_argument("--out", required=True, help="выходной .npz (сводка с CI)")
    parser.add_argument(
        "--out-per-sample",
        default="",
        help="опциональный .npz с по-сэмпльными метриками (десятки МБ; для переанализа)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="число бутстрап-ресэмплов")
    parser.add_argument("--seed", type=int, default=0, help="сид бутстрапа")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()
