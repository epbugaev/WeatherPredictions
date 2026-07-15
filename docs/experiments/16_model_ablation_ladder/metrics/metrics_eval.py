"""Полный набор метрик одного арма exp16 (PI-IAM4VP) по 12-шаговому rollout → npz.

Всё, кроме способа прогонять модель, живёт в :mod:`metrics_core` (общее ядро с exp20):
метрики по (сэмпл × шаг × канал), бутстрап-CI и схема npz — см. его docstring.
Здесь — специфика IAM4VP: авторегрессия снаружи модели (``model(x, pred_list, t)``,
каждый следующий кадр строится из уже спрогнозированных) и правило горизонта — нативные
6 шагов раскатываются до 12, чтобы диагностика была общей с exp20.

Широты берутся настоящие (16.875..63.28°N, юг→север) — см. `metrics_lib`. ACC
считается от климатологии train-лет (`climatology.py`), валидационный год в неё не
входит. Пороги CSI/FSS — квантили ИСТИНЫ, общие для всех армов.

Запуск (кластер):
    REPO_ROOT=~/wt_fix_v2 python metrics_eval.py \
        --checkpoint ~/abl16_long_ckpt/<arm>/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2000_2004.dat \
        --climatology ~/era5_memmap/climatology_usa_2000_2003.npz \
        --thresholds ~/era5_memmap/thresholds_usa_2004.npz \
        --out ~/abl16_metrics/metrics_<arm>.npz
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16_metrics_core", HERE / "metrics_core.py")
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core  # dataclass ядра резолвит аннотации через sys.modules
_spec.loader.exec_module(core)

HORIZON = 6  # дефолтный нативный горизонт; t≥12 читается из конфига


def predict_window(model, frames: torch.Tensor, horizon: int) -> torch.Tensor:
    """Нативный авторегрессивный прогон: ``horizon`` кадров от собственных прогнозов.

    Args:
        model: IAM4VP-семейство: ``model(frames, pred_list, t) -> (B, C, H, W)``.
        frames: нормированный контекст ``(B, T_ctx, C, H, W)``.
        horizon: сколько кадров прогнозировать.

    Returns:
        ``torch.Tensor`` формы ``(B, horizon, C, H, W)`` в нормированном пространстве.
    """
    pred_list: list[torch.Tensor] = []
    for idx_time in range(horizon):
        t = torch.full((frames.shape[0],), (idx_time + 1) * 100.0, device=frames.device)
        pred_list.append(model(frames, pred_list, t))
    return torch.stack(pred_list, dim=1)


def main() -> None:
    """CLI → метрики арма exp16. Side effects: пишет npz (см. `metrics_core.evaluate_arm`)."""
    args = core.parse_args(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    arm_checkpoint = core.load_arm(args.checkpoint, device)
    training_cfg = arm_checkpoint.config["training"]
    native_horizon = int(training_cfg.get("extra_kwargs", {}).get("time_prediction", HORIZON))
    # Ниже 12 шагов диагностика не опускается: армы с нативными 6 раскатываются вдвое.
    rollout_steps = native_horizon if native_horizon >= 2 * HORIZON else 2 * native_horizon

    model = arm_checkpoint.model
    core.evaluate_arm(
        arm_checkpoint=arm_checkpoint,
        forecast=lambda x, y: predict_window(model, x, y.shape[1]),
        rollout_steps=rollout_steps,
        native_horizon=native_horizon,
        args=args,
    )


if __name__ == "__main__":
    main()
