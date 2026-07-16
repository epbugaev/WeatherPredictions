"""Метрики арма IAM4VP exp22 на НАТИВНОМ горизонте (6 кадров) → npz.

Отличие от exp16-обёртки (`16_.../metrics/metrics_eval.py`): та раскатывает арм до
``2×native`` (12 шагов), потому что армы exp16 были 12-native. Армы exp22 IAM4VP
обучены на ``time_prediction=6`` (native horizon 6, ``mask_token`` размера 6), и
авторегрессия ``predict_window`` за 6 кадров падает (``mask_token[:, 6]`` вне границ).
Поэтому здесь ``rollout_steps = native_horizon`` — оцениваем ровно то, что модель
предсказывает. IAM4VP-метрики exp22 — 6-шаговые (SimVPv2/PredRNNv2 — 12-шаговые);
кросс-семейное сравнение идёт по относительной Δ к no_physics, поэтому разный
горизонт между семействами допустим (как и разный бюджет эпох, README §2).

Ядро, схема npz и `predict_window` — переиспользуются из exp16.

Запуск (кластер):
    REPO_ROOT=~/wt_exp21 python metrics_eval_iam4vp.py \
        --checkpoint ~/wt_exp21/checkpoints/exp22-iam4vp-a2-france-s0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_france_2000_2004.dat \
        --climatology ~/era5_memmap/climatology_france_2000_2003.npz \
        --thresholds ~/era5_memmap/thresholds_france_2004.npz \
        --out ~/exp22_metrics/metrics_exp22-iam4vp-a2-france-s0.npz \
        --out-per-sample ~/exp22_metrics_raw/iam4vp_france/metrics_exp22-iam4vp-a2-france-s0_per_sample.npz
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

EXP16 = Path(REPO_ROOT) / "docs/experiments/16_model_ablation_ladder/metrics"
_eval_spec = importlib.util.spec_from_file_location("exp16_metrics_eval", EXP16 / "metrics_eval.py")
exp16_eval = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(exp16_eval)
core = exp16_eval.core


def main() -> None:
    """CLI → метрики IAM4VP-арма на нативном горизонте. Side effect: пишет npz."""
    args = core.parse_args(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    arm_checkpoint = core.load_arm(args.checkpoint, device)
    training_cfg = arm_checkpoint.config["training"]
    native_horizon = int(
        training_cfg.get("extra_kwargs", {}).get("time_prediction", exp16_eval.HORIZON)
    )
    model = arm_checkpoint.model
    core.evaluate_arm(
        arm_checkpoint=arm_checkpoint,
        forecast=lambda x, y: exp16_eval.predict_window(model, x, y.shape[1]),
        rollout_steps=native_horizon,  # НАТИВНО, без 2x-раскатки (mask_token = native)
        native_horizon=native_horizon,
        args=args,
    )


if __name__ == "__main__":
    main()
