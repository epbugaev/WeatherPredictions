"""Rollout-инференс арма exp16 из чекпоинта → npz с lat-weighted RMSE.

Горизонт определяется из конфига (``training.extra_kwargs.time_prediction``):

  * **t=6** (нативный горизонт 6) — 12 шагов скользящим окном из двух прогонов:
    окно 1 (6 реальных кадров → t+6..t+11), окно 2 free-running (собственные
    прогнозы → t+12..t+17) и окно 2 teacher-forced (реальные кадры → t+12..t+17);
  * **t=12** (нативный горизонт ≥ 12) — весь прогноз ОДНИМ нативным окном:
    free-running (собственные прогнозы) vs teacher-forced (реальные кадры в
    истории ``pred_list``), без границы окон.

Teacher-forced в обоих режимах изолирует по-шаговую ошибку модели от накопления
дрейфа rollout.

Модель, нормализация и конфиг данных восстанавливаются из чекпоинта трейнера
(`payload["model"|"normalize"|"config"]`), поэтому скрипт работает и из
дерева fix_inline_v2, и из pre-exp13 worktree (армы R2a/R2 обязаны
инферироваться кодом своего дерева): ``REPO_ROOT`` задаёт, чьи модули
импортировать, сам файл скрипта может лежать в другом дереве.

Запуск (кластер, GPU):
    REPO_ROOT=~/wt_fix_v2 python rollout_eval.py \
        --checkpoint ~/abl16_ckpt/<arm>/<run>/epoch=17-*.pt \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --out ~/abl16_rollout/rollout_<arm>.npz

Выход npz: ``rmse_free``/``rmse_forced`` — (12, 69) lat-weighted RMSE в
физических единицах по полной валидации, ``channels`` (69 имён),
``n_samples``, ``checkpoint_epoch``, ``arm``.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader  # noqa: E402

import Data  # noqa: E402, F401  (регистрирует датасеты в utils.registry)
import Models  # noqa: E402, F401  (регистрирует модели в utils.registry)
from training_strategies._index_maps import MULTIOUT_INDEX_MAP  # noqa: E402
from utils.metrics import weighted_rmse_torch_channels  # noqa: E402
from utils.normalize import WeatherNormalize  # noqa: E402
from utils.registry import get_dataset, get_model  # noqa: E402

HORIZON = 6  # дефолтный нативный горизонт (t=6); t≥12 читается из конфига


def channel_names() -> list[str]:
    """69 имён каналов в порядке индексов выходного тензора (v4-layout)."""
    by_index = {idx: name for name, idx in MULTIOUT_INDEX_MAP.items()}
    return [by_index[i] for i in range(len(by_index))]


def predict_window(model, frames: torch.Tensor, horizon: int) -> torch.Tensor:
    """Один нативный прогон модели: ``horizon`` авторегрессивных кадров.

    Args:
        model: вызываемая как ``model(x, pred_list, t)`` → ``(B, C, H, W)``.
        frames: вход ``(B, T, C, H, W)`` в нормированном пространстве.
        horizon: число шагов (контракт ``IterativeManualStep``: t=(i+1)*100).

    Returns:
        Прогнозы ``(B, horizon, C, H, W)``.
    """
    pred_list: list[torch.Tensor] = []
    for idx_time in range(horizon):
        t = torch.full((frames.shape[0],), (idx_time + 1) * 100.0, device=frames.device)
        prediction = model(frames, pred_list, t)
        pred_list.append(prediction)
    return torch.stack(pred_list, dim=1)


def rollout_two_windows(
    model, x: torch.Tensor, y: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """12-шаговый прогноз двумя окнами: free-running и teacher-forced.

    Режим для моделей с нативным горизонтом < 12 (t=6): 12 шагов собираются
    скользящим окном из двух прогонов по ``horizon`` кадров.

    Args:
        model: вызываемая как ``model(x, pred_list, t)``.
        x: реальные входные кадры ``(B, horizon, C, H, W)``.
        y: реальные таргеты ``(B, >=2*horizon, C, H, W)`` (кадры ``t+6..``).
        horizon: нативный горизонт модели (6).

    Returns:
        ``(free, forced)`` формы ``(B, 2*horizon, C, H, W)``: первые
        ``horizon`` шагов общие (окно 1), дальше free продолжает от
        собственных прогнозов, forced — от реальных кадров ``y[:, :horizon]``.
    """
    window1 = predict_window(model, x, horizon)
    free2 = predict_window(model, window1, horizon)
    # y[:, :horizon] — non-contiguous view; IAM4VP.forward требует contiguous
    forced2 = predict_window(model, y[:, :horizon].contiguous(), horizon)
    free = torch.cat([window1, free2], dim=1)
    forced = torch.cat([window1, forced2], dim=1)
    return free, forced


def predict_teacher_forced(
    model, frames: torch.Tensor, truth: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Нативный прогон с teacher-forcing: в историю подаются РЕАЛЬНЫЕ кадры.

    На шаге ``i`` ``pred_list`` содержит реальные кадры ``truth[:, :i]`` (а не
    собственные прогнозы), что изолирует по-шаговую ошибку модели от накопления
    дрейфа. Шаг 0 (пустая история) совпадает с free-running.

    Args:
        model: вызываемая как ``model(frames, pred_list, t)``.
        frames: входное окно ``(B, T_in, C, H, W)`` (нормировано).
        truth: реальные таргеты ``(B, horizon, C, H, W)`` — учитель.
        horizon: число авторегрессивных шагов.

    Returns:
        Прогнозы ``(B, horizon, C, H, W)``.
    """
    pred_list: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    for idx_time in range(horizon):
        t = torch.full((frames.shape[0],), (idx_time + 1) * 100.0, device=frames.device)
        preds.append(model(frames, pred_list, t))
        pred_list.append(truth[:, idx_time])
    return torch.stack(preds, dim=1)


def rollout_native(
    model, x: torch.Tensor, y: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Нативный горизонт ≥ 12 (t=12): одно окно, free-running vs teacher-forced.

    Args:
        model: вызываемая как ``model(x, pred_list, t)``.
        x: входное окно ``(B, T_in, C, H, W)`` (12 реальных кадров).
        y: реальные таргеты ``(B, horizon, C, H, W)``.
        horizon: нативный горизонт (12) — весь прогноз одним окном.

    Returns:
        ``(free, forced)`` формы ``(B, horizon, C, H, W)``: free авторегрессивен
        от собственных прогнозов, forced — с реальной историей (общий шаг 1).
    """
    free = predict_window(model, x, horizon)
    forced = predict_teacher_forced(model, x, y, horizon)
    return free, forced


def build_val_dataset(data_cfg: dict, memmap_path: str, rollout_steps: int):
    """Вал-датасет арма с таргетом, удлинённым до ``rollout_steps`` кадров.

    Повторяет ``train.build_dataset(data_cfg, "val")`` для v4-memmap, меняя
    только ``end_time_y`` (11 → 17 при 12 шагах) и путь memmap.

    Args:
        data_cfg: блок ``data`` конфига из чекпоинта.
        memmap_path: путь к packed-memmap ``.dat`` (meta — рядом,
            ``<stem>.meta.json``).
        rollout_steps: полное число шагов rollout (2 окна × horizon).

    Returns:
        Датасет: сэмпл ``(x (6,C,H,W), y (rollout_steps,C,H,W))``, raw float32.
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


def evaluate_checkpoint(args: Namespace) -> None:
    """Прогнать полный вал через 2-оконный rollout и записать npz.

    Side effects: пишет ``args.out`` (npz), печатает прогресс и sanity-строки
    (RMSE z500/t850 шагов 1 и 6 — для сверки с Comet ``RMSE_*_first/last``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["config"]
    arm = cfg.get("experiment", {}).get("name", Path(args.checkpoint).parent.name)

    model = get_model(cfg["model"]["type"])(**cfg["model"].get("params", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    n_channels = len(channel_names())
    normalize = WeatherNormalize(mean=torch.zeros(n_channels), std=torch.ones(n_channels))
    normalize.load_state_dict(payload["normalize"])
    normalize.to(device)

    set_physics_normalization = getattr(model, "set_physics_normalization", None)
    if set_physics_normalization is not None:
        set_physics_normalization(normalize.mean.view(-1).cpu(), normalize.std.view(-1).cpu())

    # Нативный горизонт из конфига: t=6 → скользящие 2 окна (12 шагов);
    # t≥12 → одно нативное окно (free vs teacher-forced реальными кадрами).
    native_horizon = int(cfg["training"].get("extra_kwargs", {}).get("time_prediction", HORIZON))
    single_window = native_horizon >= 2 * HORIZON
    rollout_steps = native_horizon if single_window else 2 * native_horizon

    dataset = build_val_dataset(cfg["data"], args.memmap, rollout_steps)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    std_phys = normalize.std.view(-1)  # (C,) для перевода RMSE в физединицы
    rmse_free_sum = torch.zeros(rollout_steps, n_channels, device=device)
    rmse_forced_sum = torch.zeros(rollout_steps, n_channels, device=device)
    n_samples = 0

    with torch.inference_mode():
        for batch_idx, (x_raw, y_raw) in enumerate(loader):
            x = normalize(x_raw.to(device, non_blocking=True))
            y = normalize(y_raw.to(device, non_blocking=True))
            if single_window:
                free, forced = rollout_native(model, x, y, native_horizon)
            else:
                free, forced = rollout_two_windows(model, x, y, native_horizon)
            # (B, T, C) в нормированных единицах → физединицы через std
            rmse_free_sum += (weighted_rmse_torch_channels(free, y) * std_phys).sum(dim=0)
            rmse_forced_sum += (weighted_rmse_torch_channels(forced, y) * std_phys).sum(dim=0)
            n_samples += x.shape[0]
            if batch_idx % 10 == 0:
                print(f"[rollout] {arm}: batch {batch_idx}, samples {n_samples}")  # noqa: T201
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break

    rmse_free = (rmse_free_sum / n_samples).cpu().numpy()
    rmse_forced = (rmse_forced_sum / n_samples).cpu().numpy()

    names = channel_names()
    z500, t850 = names.index("z500"), names.index("t850")
    print(  # noqa: T201
        f"[sanity] {arm}: z500 step1={rmse_free[0, z500]:.2f} "
        f"step6={rmse_free[5, z500]:.2f}; "
        f"t850 step1={rmse_free[0, t850]:.3f} step6={rmse_free[5, t850]:.3f}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        rmse_free=rmse_free,
        rmse_forced=rmse_forced,
        channels=np.array(names),
        n_samples=n_samples,
        checkpoint_epoch=int(payload.get("epoch", -1)),
        native_horizon=native_horizon,
        arm=arm,
    )
    print(f"[rollout] {arm}: written {out_path} (n={n_samples})")  # noqa: T201


def parse_args() -> Namespace:
    """CLI: чекпоинт, memmap, выходной npz, параметры загрузчика."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="путь к .pt чекпоинту")
    parser.add_argument("--memmap", required=True, help="packed memmap .dat")
    parser.add_argument("--out", required=True, help="выходной .npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="ограничить число батчей (смоук); 0 = вся валидация",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
