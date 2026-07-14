"""Полный набор метрик одного арма exp20 (PI-PredRNNv2) по 12-шаговому rollout → npz.

Ядро — общее с exp16: :mod:`metrics_core` из
``docs/experiments/16_model_ablation_ladder/metrics/`` (метрики по
(сэмпл × шаг × канал), парный бутстрап и **схема npz**, которую дальше без изменений
читают ``paired_deltas.py`` и ``metrics_figures.py``). Здесь — единственное, чем
архитектуры отличаются: как модель гонится по окну прогноза.

PredRNN авторегрессирует **внутри** своего ``forward``: вход — склейка контекста и
таргета в channels-last, второй аргумент — маска scheduled sampling (нули = модель
кормит себя собственными прогнозами; так же валидируется обучение). Поэтому 12 шагов
берутся ОДНИМ форвардом — скользящее окно exp16 (нативные 6) здесь не нужно, — а
срез окна прогноза (``x.shape[1] - 1``: PredRNN отдаёт ``T-1`` кадров, где элемент
``i`` предсказывает кадр ``i+1``) живёт в :func:`training_strategies.predrnn.predrnn_forecast`,
общей с валидацией. Расхождение здесь молча сдвинуло бы все лид-тайм на кадр.

Чекпоинт волны — ``last.pt`` (эпоха 40, общая для всех шести армов): ранжировать армы
можно только на ОБЩЕЙ эпохе (exp16 §11.3 — разброс эпох ``best.pt`` даёт больше
скилла, чем измеряемый эффект физики).

Запуск (кластер, CPU-нода):
    REPO_ROOT=~/wt_exp20 python metrics_eval.py \
        --checkpoint ~/exp20_ckpt/exp20-p3-a2-exp13-s0/<run>/last.pt \
        --memmap ~/era5_memmap/predformer_usa_2004.dat \
        --climatology ~/era5_memmap/climatology_usa_2000_2003.npz \
        --thresholds ~/era5_memmap/thresholds_usa_2004.npz \
        --out ~/exp20_metrics/metrics_exp20-p3-a2-exp13-s0.npz \
        --out-per-sample ~/exp20_metrics_raw/metrics_exp20-p3-a2-exp13-s0_per_sample.npz
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, REPO_ROOT)

from training_strategies.predrnn import predrnn_forecast  # noqa: E402

EXP16_METRICS = Path(REPO_ROOT) / "docs/experiments/16_model_ablation_ladder/metrics"
_spec = importlib.util.spec_from_file_location(
    "exp16_metrics_core", EXP16_METRICS / "metrics_core.py"
)
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core  # dataclass ядра резолвит аннотации через sys.modules
_spec.loader.exec_module(core)


def main() -> None:
    """CLI → метрики арма exp20. Side effects: пишет npz (см. `metrics_core.evaluate_arm`)."""
    args = core.parse_args(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    arm_checkpoint = core.load_arm(args.checkpoint, device)
    # Горизонт задаёт сама модель: сколько кадров она прогнозирует за один форвард.
    horizon = int(arm_checkpoint.config["model"]["params"]["configs"]["aft_seq_length"])

    model = arm_checkpoint.model
    core.evaluate_arm(
        arm_checkpoint=arm_checkpoint,
        forecast=lambda x, y: predrnn_forecast(model, x, y, teacher_forcing=False),
        rollout_steps=horizon,
        native_horizon=horizon,
        args=args,
    )


if __name__ == "__main__":
    main()
