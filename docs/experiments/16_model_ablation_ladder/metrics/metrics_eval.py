"""Полный набор метрик одного арма exp16 по 12-шаговому rollout → npz.

Считает по каждому (сэмпл × шаг × канал), чтобы бутстрап-CI строился снаружи:

  * ``rmse``, ``acc``, ``bias``, ``w1`` — по-сэмпльно ``(S, 12, 69)``;
  * ``std_pred`` / ``std_obs`` — пространственный std поля (проверка на сглаживание);
  * ``csi_tp/fp/fn`` — взвешенные счётчики на пороги p90/p95/p99 ``(S, 12, 69, 3)``
    (агрегировать надо счётчики, а не готовый CSI: сэмплы без событий его ломают);
  * ``fss_num/den`` на те же пороги и окна 1/3/5 ячеек ``(S, 12, 69, 3, 3)``;
  * ``psd_pred`` / ``psd_obs`` — спектр по зональному волновому числу m ``(12, 69, 33)``.

Широты берутся настоящие (16.875..63.28°N, юг→север) — см. `metrics_lib`. ACC
считается от климатологии train-лет (`climatology.py`), валидационный год в неё не
входит. Пороги CSI/FSS — квантили ИСТИНЫ, общие для всех армов.

Запуск (кластер):
    REPO_ROOT=~/wt_fix_v2 python metrics_eval.py \
        --checkpoint ~/abl16_long_ckpt/<arm>/<run>/best.pt \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --climatology ~/era5_memmap/climatology_usa_2000_2003.npz \
        --out ~/abl16_metrics/metrics_<arm>.npz
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader  # noqa: E402

import Data  # noqa: E402, F401  (регистрирует датасеты в utils.registry)
import Models  # noqa: E402, F401  (регистрирует модели в utils.registry)
from training_strategies._index_maps import MULTIOUT_INDEX_MAP  # noqa: E402
from utils.normalize import WeatherNormalize  # noqa: E402
from utils.registry import get_dataset, get_model  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16_metrics_lib", HERE / "metrics_lib.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

QUANTILES = (0.90, 0.95, 0.99)  # пороги событий CSI/FSS (по истине, общие для армов)
NEIGHBORHOODS = (1, 3, 5)  # окна FSS в ячейках (1 = по-ячеечно)
HORIZON = 6  # дефолтный нативный горизонт; t≥12 читается из конфига


def channel_names() -> list[str]:
    """69 имён каналов в порядке индексов выходного тензора (v4-layout)."""
    by_index = {idx: name for name, idx in MULTIOUT_INDEX_MAP.items()}
    return [by_index[i] for i in range(len(by_index))]


def predict_window(model, frames: torch.Tensor, horizon: int) -> torch.Tensor:
    """Нативный авторегрессивный прогон: ``horizon`` кадров от собственных прогнозов."""
    pred_list: list[torch.Tensor] = []
    for idx_time in range(horizon):
        t = torch.full((frames.shape[0],), (idx_time + 1) * 100.0, device=frames.device)
        pred_list.append(model(frames, pred_list, t))
    return torch.stack(pred_list, dim=1)


def build_val_dataset(data_cfg: dict, memmap_path: str, rollout_steps: int):
    """Вал-датасет арма с таргетом, удлинённым до ``rollout_steps`` кадров.

    Повторяет ``rollout_eval.build_val_dataset``: тот же v4-memmap, меняется только
    ``end_time_y`` и путь. Датасет нужен ещё и как источник времени: климатология для
    ACC ищется по дню года каждого таргет-кадра.
    """
    split_cfg = data_cfg.get("val", {})
    start_time_y = int(data_cfg.get("start_time_y", 6))
    params = {
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


def target_day_of_year(dataset, rollout_steps: int, start_time_y: int) -> np.ndarray:
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


def evaluate(args: Namespace) -> None:
    """Прогнать валидацию, посчитать все метрики и записать npz.

    Side effects: пишет ``args.out``, печатает прогресс и sanity-строку.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["config"]
    arm = cfg.get("experiment", {}).get("name", Path(args.checkpoint).parent.name)

    model = get_model(cfg["model"]["type"])(**cfg["model"].get("params", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    names = channel_names()
    n_channels = len(names)
    normalize = WeatherNormalize(mean=torch.zeros(n_channels), std=torch.ones(n_channels))
    normalize.load_state_dict(payload["normalize"])
    normalize.to(device)
    set_physics_normalization = getattr(model, "set_physics_normalization", None)
    if set_physics_normalization is not None:
        set_physics_normalization(normalize.mean.view(-1).cpu(), normalize.std.view(-1).cpu())

    native_horizon = int(cfg["training"].get("extra_kwargs", {}).get("time_prediction", HORIZON))
    rollout_steps = native_horizon if native_horizon >= 2 * HORIZON else 2 * native_horizon
    start_time_y = int(cfg["data"].get("start_time_y", 6))

    dataset = build_val_dataset(cfg["data"], args.memmap, rollout_steps)
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
            obs = y_raw.to(device, non_blocking=True)[:, :rollout_steps]  # физ. единицы
            pred_norm = predict_window(model, x, rollout_steps)
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
        f"ACC t850 шаг12={raw['acc'][:, -1, t850].mean():.3f}"
    )

    # Бутстрап делаем здесь: по-сэмпльные массивы весят десятки МБ на арм и в
    # репозиторий не поедут, а сводка — сотни КБ.
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
    fss_ratio = ml.bootstrap_ratio_ci(num, den, args.bootstrap, seed=args.seed)
    summary |= {
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
        checkpoint_epoch=int(payload.get("epoch", -1)),
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


def parse_args() -> Namespace:
    """CLI: чекпоинт, memmap, климатология, пороги, выходной npz."""
    parser = ArgumentParser(description=__doc__)
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


if __name__ == "__main__":
    evaluate(parse_args())
