"""Полный набор метрик одного арма exp21 (PI-SimVPv2) по 12-шаговому прогнозу → npz.

Ядро — общее с exp16/exp20: :mod:`metrics_core` из
``docs/experiments/16_model_ablation_ladder/metrics/`` (метрики по
(сэмпл × шаг × канал), парный бутстрап и **схема npz**, которую дальше без изменений
читают ``paired_deltas.py`` и ``metrics_figures.py``). Здесь — единственное, чем
архитектура отличается: как модель гонится по окну прогноза.

SimVPv2 — MIMO: весь горизонт выдаётся ОДНИМ форвардом ``model(x) -> (B, T, C, H, W)``,
рекуррентности и скользящего окна нет (в отличие от авторегрессии IAM4VP и маски
PredRNN). Поэтому ``forecast(x, y)`` = ``model(x)`` — самый простой из трёх носителей.
Физическая коррекция уже внутри ``forward`` (режим ``physics_coupling`` арма), так что
метрики видят ровно то состояние, что и валидация.

Чекпоинт волны — ``last.pt`` (эпоха 500, общая для всех восьми армов): ранжировать
армы можно только на ОБЩЕЙ эпохе (exp16 §11.3 — разброс эпох ``best.pt`` даёт больше
скилла, чем измеряемый эффект физики).

Запуск (кластер, CPU-нода):
    REPO_ROOT=~/wt_exp21 python metrics_eval.py \
        --checkpoint ~/exp21_ckpt/exp21L-s3c-a2-exp13-chained-t12-seed0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2004.dat \
        --climatology ~/era5_memmap/climatology_usa_2000_2003.npz \
        --thresholds ~/era5_memmap/thresholds_usa_2004.npz \
        --out ~/exp21_metrics/metrics_exp21L-s3c-a2-exp13-chained.npz \
        --out-per-sample ~/exp21_metrics_raw/metrics_exp21L-s3c-a2-exp13-chained_per_sample.npz
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

EXP16_METRICS = Path(REPO_ROOT) / "docs/experiments/16_model_ablation_ladder/metrics"
_spec = importlib.util.spec_from_file_location(
    "exp16_metrics_core", EXP16_METRICS / "metrics_core.py"
)
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core  # dataclass ядра резолвит аннотации через sys.modules
_spec.loader.exec_module(core)


def main() -> None:
    """CLI → метрики арма exp21. Side effects: пишет npz (см. `metrics_core.evaluate_arm`)."""
    args = core.parse_args(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    arm_checkpoint = core.load_arm(args.checkpoint, device)
    # Горизонт задаёт сама модель: длина входного клипа = число прогнозируемых кадров.
    horizon = int(arm_checkpoint.config["model"]["params"]["in_shape"][0])

    model = arm_checkpoint.model
    core.evaluate_arm(
        arm_checkpoint=arm_checkpoint,
        forecast=lambda x, y: model(x)[:, : y.shape[1]],
        rollout_steps=horizon,
        native_horizon=horizon,
        args=args,
    )


if __name__ == "__main__":
    main()
