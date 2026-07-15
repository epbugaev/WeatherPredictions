"""Rollout-инференс арма exp20 (PI-PredRNNv2) из чекпоинта → npz с lat-weighted RMSE.

Нативный горизонт PredRNNv2 в exp20 равен горизонту диагностики (12:
``pre_seq_length: 12``, ``aft_seq_length: 12``), поэтому все шаги берутся ОДНИМ
форвардом — скользящее окно exp16 (нативные 6 шагов) здесь не нужно. Оба режима
даёт сама модель через маску scheduled sampling, авторегрессия живёт внутри её
``forward``:

  * ``mask_true = 0`` → после контекстных кадров модель кормит себя
    собственными прогнозами (**free-running**; ровно то, что делает валидация
    в :mod:`training_strategies.predrnn`);
  * ``mask_true = 1`` → на каждом шаге в неё входит реальный кадр
    (**teacher-forced**) — изолирует по-шаговую ошибку от накопления дрейфа.

Шаг 1 у обоих режимов общий: первый прогноз делается из последнего РЕАЛЬНОГО
кадра контекста. Срез окна прогноза (``x.shape[1] - 1``) живёт в
:func:`training_strategies.predrnn.predrnn_forecast` — общий с валидацией,
чтобы диагностика и обучение не разъехались на кадр.

Чекпоинты волны — ``last.pt`` (эпоха 40, общая для всех армов) и ``best.pt``:
per-epoch копии трейнер больше не пишет, путь передаётся напрямую.

Запуск (кластер, CPU-нода):
    python rollout_eval.py \
        --checkpoint ~/exp20_ckpt/exp20-p3-a2-exp13-s0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2004.dat \
        --out ~/exp20_rollout/rollout_exp20-p3-a2-exp13-s0.npz

Выход npz (схема общая с exp16 → фигуры переиспользуются): ``rmse_free`` /
``rmse_forced`` формы ``(12, 69)`` — lat-weighted RMSE в физических единицах по
полной валидации 2004, ``channels``, ``n_samples``, ``checkpoint_epoch``,
``native_horizon``, ``arm``.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader  # noqa: E402

from tools.rollout_common import (  # noqa: E402
    LatWeightedRmseAccumulator,
    build_val_dataset,
    channel_names,
    load_arm_checkpoint,
    save_rollout_npz,
)
from training_strategies.predrnn import predrnn_forecast  # noqa: E402


def evaluate_checkpoint(args: Namespace) -> None:
    """Прогнать полную валидацию в обоих режимах rollout и записать npz.

    Args:
        args: разобранный CLI (см. :func:`parse_args`).

    Side effects: пишет ``args.out`` (npz), печатает прогресс и sanity-строку
    (RMSE z500/t850 на шагах 1 и 12 — для сверки с Comet ``RMSE_*_first/last``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    arm_checkpoint = load_arm_checkpoint(args.checkpoint, device)
    model, normalize = arm_checkpoint.model, arm_checkpoint.normalize
    arm, config = arm_checkpoint.arm, arm_checkpoint.config

    # Горизонт задан самой моделью: сколько кадров она прогнозирует за форвард.
    native_horizon = int(config["model"]["params"]["configs"]["aft_seq_length"])

    dataset = build_val_dataset(config["data"], args.memmap, native_horizon)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    std = normalize.std.reshape(-1)
    free_rmse = LatWeightedRmseAccumulator(native_horizon, std, device)
    forced_rmse = LatWeightedRmseAccumulator(native_horizon, std, device)

    with torch.inference_mode():
        for batch_index, (x_raw, y_raw) in enumerate(loader):
            x = normalize(x_raw.to(device, non_blocking=True))
            y = normalize(y_raw.to(device, non_blocking=True))
            free_rmse.update(predrnn_forecast(model, x, y, teacher_forcing=False), y)
            forced_rmse.update(predrnn_forecast(model, x, y, teacher_forcing=True), y)
            if batch_index % 10 == 0:
                print(  # noqa: T201
                    f"[rollout] {arm}: batch {batch_index}, samples {free_rmse.n_samples}"
                )
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break

    rmse_free, rmse_forced = free_rmse.mean(), forced_rmse.mean()
    names = channel_names()
    z500, t850 = names.index("z500"), names.index("t850")
    print(  # noqa: T201
        f"[sanity] {arm}: z500 step1={rmse_free[0, z500]:.2f} "
        f"step{native_horizon}={rmse_free[-1, z500]:.2f}; "
        f"t850 step1={rmse_free[0, t850]:.3f} step{native_horizon}={rmse_free[-1, t850]:.3f}"
    )

    save_rollout_npz(
        out_path=args.out,
        rmse_free=rmse_free,
        rmse_forced=rmse_forced,
        n_samples=free_rmse.n_samples,
        checkpoint_epoch=arm_checkpoint.epoch,
        native_horizon=native_horizon,
        arm=arm,
    )
    print(f"[rollout] {arm}: written {args.out} (n={free_rmse.n_samples})")  # noqa: T201


def parse_args() -> Namespace:
    """CLI: чекпоинт, memmap, выходной npz, параметры загрузчика."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="путь к .pt (last.pt / best.pt)")
    parser.add_argument("--memmap", required=True, help="packed memmap .dat")
    parser.add_argument("--out", required=True, help="выходной .npz")
    parser.add_argument("--batch-size", type=int, default=8)
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
