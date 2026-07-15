"""Rollout-инференс арма exp21 (PI-SimVPv2) из чекпоинта → npz с lat-weighted RMSE.

SimVPv2 — MIMO: весь горизонт (12 кадров) выдаётся ОДНИМ форвардом
``model(x) -> (B, T, C, H, W)``, авторегрессии и скользящего окна нет. Поэтому,
в отличие от exp16 (IAM4VP, окно 6) и в отличие от «двух режимов» exp20:

  * **free-running** и **teacher-forced совпадают** — модель не потребляет ни
    своих, ни истинных промежуточных кадров, а отображает контекст сразу во все
    выходные кадры. Значит ``rmse_forced == rmse_free``, и во второй режим писать
    нечего; фигуры (``single_mode`` в :class:`tools.rollout_ladder.RolloutSpec`)
    рисуют одну панель, а не вырожденную пару.

Схема npz — общая с exp16/exp20 (:mod:`tools.rollout_common`), чтобы фигуры
переиспользовались: ``rmse_free``/``rmse_forced`` формы ``(12, 69)`` в физических
единицах, ``channels``, ``n_samples``, ``checkpoint_epoch``, ``native_horizon``,
``arm``. Чекпоинт волны — ``last.pt`` (эпоха 500, общая для всех армов): ранжировать
армы можно только на ОБЩЕЙ эпохе (exp16 §11.3).

Запуск (кластер, CPU-нода):
    REPO_ROOT=~/wt_exp21 python rollout_eval.py \
        --checkpoint ~/exp21_ckpt/exp21L-s3c-a2-exp13-chained-t12-seed0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2004.dat \
        --out ~/exp21_rollout/rollout_exp21L-s3c-a2-exp13-chained-t12-seed0.npz
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


def evaluate_checkpoint(args: Namespace) -> None:
    """Прогнать полную валидацию одним форвардом MIMO и записать npz.

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

    # Горизонт задан самой моделью: длина входного клипа = число прогнозируемых кадров.
    native_horizon = int(config["model"]["params"]["in_shape"][0])

    dataset = build_val_dataset(config["data"], args.memmap, native_horizon)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    std = normalize.std.reshape(-1)
    rmse = LatWeightedRmseAccumulator(native_horizon, std, device)

    with torch.inference_mode():
        for batch_index, (x_raw, y_raw) in enumerate(loader):
            x = normalize(x_raw.to(device, non_blocking=True))
            y = normalize(y_raw.to(device, non_blocking=True))
            rmse.update(model(x)[:, : y.shape[1]], y)
            if batch_index % 10 == 0:
                print(f"[rollout] {arm}: batch {batch_index}, samples {rmse.n_samples}")  # noqa: T201
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break

    # MIMO: free-running и teacher-forced совпадают (нет подачи промежуточных кадров).
    rmse_free = rmse.mean()
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
        rmse_forced=rmse_free,
        n_samples=rmse.n_samples,
        checkpoint_epoch=arm_checkpoint.epoch,
        native_horizon=native_horizon,
        arm=arm,
    )
    print(f"[rollout] {arm}: written {args.out} (n={rmse.n_samples})")  # noqa: T201


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
