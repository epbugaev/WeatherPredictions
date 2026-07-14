#!/bin/bash
#SBATCH --job-name=exp20-metrics
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик (RMSE/ACC/bias/W1/CSI/FSS/std/PSD) всех армов лестницы exp20
# (PI-PredRNNv2) с общего чекпоинта last.pt (эпоха 40) + парные бутстрап-дельты к P0.
#
# Выход: сводки с CI в ${OUT_DIR} и по-сэмпльные npz в ${RAW_DIR} (десятки МБ на арм —
# нужны парному бутстрапу и остаются на кластере), затем ${OUT_DIR}/paired_deltas.npz.
# Локально по ним рисуются фигуры: docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_figures.py.
#
# Считаем на CPU: горизонт нативный (12 шагов = один форвард), а cpu-e-quick
# несравнимо менее загружен, чем GPU-очереди.
set -euo pipefail
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp20}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT_BASE="${CKPT_BASE:-${HOME}/exp20_ckpt}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2004.dat}"
OUT_DIR="${OUT_DIR:-${HOME}/exp20_metrics}"
RAW_DIR="${RAW_DIR:-${HOME}/exp20_metrics_raw}"
CKPT_NAME="${CKPT_NAME:-last.pt}"

# Климатология (база отсчёта ACC) и пороги CSI/FSS (квантили ИСТИНЫ) переиспользуются
# от exp16: те же данные, тот же USA-кроп и тот же валидационный год (2004), а
# климатология считана по train-годам 2000-2003, поэтому валидация в неё не подсматривает.
# Общие пороги на все армы обязательны — иначе армы сравнивались бы на разных событиях.
CLIMATOLOGY="${CLIMATOLOGY:-${HOME}/era5_memmap/climatology_usa_2000_2003.npz}"
THRESHOLDS="${THRESHOLDS:-${HOME}/era5_memmap/thresholds_usa_2004.npz}"

EXP16_METRICS="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics"
SCRIPT="${REPO_ROOT}/docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_eval.py"
BASELINE="exp20-p0-no-physics"  # канонический ключ контроля (суффикс -s0 срезается)
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

if [[ ! -f "${CLIMATOLOGY}" ]]; then
  echo "[exp20-metrics] нет климатологии ${CLIMATOLOGY}; построить: python ${EXP16_METRICS}/climatology.py" >&2
  exit 2
fi
if [[ ! -f "${THRESHOLDS}" ]]; then
  echo "[exp20-metrics] нет порогов ${THRESHOLDS} -> считаю по валидации 2004"
  python "${EXP16_METRICS}/thresholds.py" \
    --memmap "${MEMMAP}" \
    --out "${THRESHOLDS}" \
    --val-year 2004
fi

# Порядок = порядок лестницы (README §4): контроль, легаси, затем A-семейство.
# EXP20_ARMS позволяет считать подмножество (армы финишируют вразнобой, а гонять
# уже готовые можно, не дожидаясь отстающих). ВНИМАНИЕ: в общий парный бутстрап
# должны попасть только армы с чекпоинтом ОДНОЙ эпохи — сравнение арм на разных
# эпохах невалидно (exp16 §11.3: разброс эпох стоит больше, чем эффект физики).
read -r -a ARMS <<<"${EXP20_ARMS:-p0_no_physics p1_legacy_hybrid p3a_no_diabatic p3_a2_exp13 p4_exp14 p5_exp15}"

for arm in "${ARMS[@]}"; do
  run_name="exp20-${arm//_/-}-s0"
  ckpt="$(ls -t "${CKPT_BASE}/${run_name}"/*/"${CKPT_NAME}" | head -1)"
  echo "[exp20-metrics] arm=${run_name} ckpt=${ckpt}"
  python "${SCRIPT}" \
    --checkpoint "${ckpt}" \
    --memmap "${MEMMAP}" \
    --climatology "${CLIMATOLOGY}" \
    --thresholds "${THRESHOLDS}" \
    --out "${OUT_DIR}/metrics_${run_name}.npz" \
    --out-per-sample "${RAW_DIR}/metrics_${run_name}_per_sample.npz" \
    --batch-size 8 \
    --num-workers 8
done

# Парные дельты: один набор бутстрап-индексов на все армы (они прогнаны по ОДНИМ И ТЕМ
# ЖЕ сэмплам, поэтому независимые CI переоценили бы неопределённость разницы).
python "${EXP16_METRICS}/paired_deltas.py" \
  --per-sample-dir "${RAW_DIR}" \
  --baseline "${BASELINE}" \
  --out "${OUT_DIR}/paired_deltas.npz"

echo "[exp20-metrics] all arms done -> ${OUT_DIR} (сводки + paired_deltas.npz), ${RAW_DIR} (по-сэмпльно)"
