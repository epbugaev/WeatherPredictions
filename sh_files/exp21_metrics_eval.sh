#!/bin/bash
#SBATCH --job-name=exp21-metrics
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик (RMSE/ACC/bias/W1/CSI/FSS/std/PSD) всех армов лестницы exp21
# (PI-SimVPv2) с общего чекпоинта last.pt (эпоха 500) + парные бутстрап-дельты к S0.
#
# Выход: сводки с CI в ${OUT_DIR} и по-сэмпльные npz в ${RAW_DIR} (десятки МБ на арм —
# нужны парному бутстрапу и остаются на кластере), затем ${OUT_DIR}/paired_deltas.npz.
# Локально по ним рисуются фигуры: docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_figures.py.
#
# Считаем на CPU: горизонт нативный (12 кадров = один форвард MIMO), а cpu-e-quick
# несравнимо менее загружен, чем GPU-очереди.
set -euo pipefail
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp21}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT_BASE="${CKPT_BASE:-${HOME}/exp21_ckpt}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2004.dat}"
OUT_DIR="${OUT_DIR:-${HOME}/exp21_metrics}"
RAW_DIR="${RAW_DIR:-${HOME}/exp21_metrics_raw}"
CKPT_NAME="${CKPT_NAME:-last.pt}"

# Климатология (база отсчёта ACC) и пороги CSI/FSS (квантили ИСТИНЫ) переиспользуются
# от exp16/exp20: те же данные, тот же USA-кроп и тот же валидационный год (2004), а
# климатология считана по train-годам 2000-2003, поэтому валидация в неё не подсматривает.
# Общие пороги на все армы обязательны — иначе армы сравнивались бы на разных событиях.
CLIMATOLOGY="${CLIMATOLOGY:-${HOME}/era5_memmap/climatology_usa_2000_2003.npz}"
THRESHOLDS="${THRESHOLDS:-${HOME}/era5_memmap/thresholds_usa_2004.npz}"

EXP16_METRICS="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics"
SCRIPT="${REPO_ROOT}/docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_eval.py"
BASELINE="exp21L-s0-no-physics"  # канонический ключ контроля (суффикс -t12-seed0 срезается)
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

if [[ ! -f "${CLIMATOLOGY}" ]]; then
  echo "[exp21-metrics] нет климатологии ${CLIMATOLOGY}; построить: python ${EXP16_METRICS}/climatology.py" >&2
  exit 2
fi
if [[ ! -f "${THRESHOLDS}" ]]; then
  echo "[exp21-metrics] нет порогов ${THRESHOLDS} -> считаю по валидации 2004"
  python "${EXP16_METRICS}/thresholds.py" \
    --memmap "${MEMMAP}" \
    --out "${THRESHOLDS}" \
    --val-year 2004
fi

# Порядок = порядок лестницы (README §4): baseline, легаси, A-семейство (batched),
# затем контроль связки chained (S0c без физики, S3c с физикой).
# EXP21_ARMS позволяет считать подмножество (армы финишируют вразнобой). ВНИМАНИЕ:
# в общий парный бутстрап должны попасть только армы с чекпоинтом ОДНОЙ эпохи —
# сравнение арм на разных эпохах невалидно (exp16 §11.3).
read -r -a ARMS <<<"${EXP21_ARMS:-s0_no_physics s1_legacy_hybrid s3a_no_diabatic s3_a2_exp13 s4_exp14 s5_exp15 s0c_no_physics_chained s3c_a2_exp13_chained}"

for arm in "${ARMS[@]}"; do
  run_name="exp21L-${arm//_/-}-t12-seed0"
  ckpt="$(ls -t "${CKPT_BASE}/${run_name}"/*/"${CKPT_NAME}" | head -1)"
  echo "[exp21-metrics] arm=${run_name} ckpt=${ckpt}"
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
#
# У SimVPv2 контроль «без физики» ДВА (§8.5): S0 = batched, S0c = chained. Связка
# `chained` двигает скилл на порядок сильнее физики, поэтому меряем физику к контролю
# СВОЕЙ связки — считаем дельты дважды (оба прогона seed 0 на 727 сэмплах → одни и те же
# бутстрап-индексы, парность сохраняется). Локально их склеивает merge_physics_baselines.py
# (batched-армы из vs-s0, S3c из vs-s0c) в paired_deltas.npz, который читают фигуры.
for base in "${BASELINE}" "exp21L-s0c-no-physics-chained"; do
  tag="${base##*exp21L-}"; tag="${tag%%-no-physics*}"; tag="${tag%%-*}"
  python "${EXP16_METRICS}/paired_deltas.py" \
    --per-sample-dir "${RAW_DIR}" \
    --baseline "${base}" \
    --out "${OUT_DIR}/paired_deltas_vs_${tag}.npz"
done

echo "[exp21-metrics] all arms done -> ${OUT_DIR} (сводки + paired_deltas_vs_{s0,s0c}.npz), ${RAW_DIR} (по-сэмпльно)"
echo "[exp21-metrics] локально: merge_physics_baselines.py --vs-s0 ... --vs-s0c ... --out paired_deltas.npz, затем metrics_figures.py"
