#!/bin/bash
#SBATCH --job-name=exp24-metrics
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик IAM4VP-армов exp24 (статические входы, USA) + парные
# бутстрап-дельты к контролю no_physics+orography. Носитель прогноза — IAM4VP
# (авторегрессия снаружи модели), поэтому eval — exp16-обёртка (12-шаговый
# rollout, родитель армов = abl16 t12). Конструирование static-модели читает
# constants-файл — запускать ТОЛЬКО на кластере (путь к constants в конфиге).
#
# Армы сравниваются на last.pt: чекпоинты стоят на РАЗНЫХ эпохах (обучение
# оборвано концом proj_1715), общей эпохи нет. Разброс эпох ≫ эффекта статики —
# см. оговорку в docs/experiments/24_static_inputs/metrics/README.md.
#
#   sbatch sh_files/exp24_metrics_eval.sh
#   EXP24_ARMS="nophys-orog a2-orog" sbatch sh_files/exp24_metrics_eval.sh   # подмножество
set -euo pipefail
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp24}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

REGION="usa"
CKPT_BASE="${CKPT_BASE:-${REPO_ROOT}/checkpoints}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
OUT_DIR="${OUT_DIR:-${HOME}/exp24_metrics}"
# Raw по-сэмпльно изолирован: paired_deltas.py глобит весь каталог, поэтому в нём
# должны лежать только армы одной лестницы.
RAW_DIR="${RAW_DIR:-${HOME}/exp24_metrics_raw/iam4vp_${REGION}}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

# Родитель IAM4VP-армов — abl16 t12: 12-шаговый rollout, eval-обёртка exp16.
EVAL="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/metrics_eval.py"

MEMMAP="${MEMMAP_DIR}/predformer_${REGION}_2000_2004.dat"
CLIMATOLOGY="${MEMMAP_DIR}/climatology_${REGION}_2000_2003.npz"
THRESHOLDS="${MEMMAP_DIR}/thresholds_${REGION}_2004.npz"
for f in "${MEMMAP}" "${CLIMATOLOGY}" "${THRESHOLDS}"; do
  if [[ ! -f "${f}" ]]; then echo "[exp24-metrics] нет ${f}" >&2; exit 2; fi
done

# Порядок: контроль nophys-orog первым (он же baseline парного бутстрапа).
read -r -a ARMS <<<"${EXP24_ARMS:-nophys-orog a2-orog nophys-static a2-static}"
for arm in "${ARMS[@]}"; do
  run="exp24-iam4vp-${arm}-${REGION}-s0"
  ckpt="$(ls -t "${CKPT_BASE}/${run}"/*/"${CKPT_NAME}" 2>/dev/null | head -1)"
  if [[ -z "${ckpt}" ]]; then echo "[exp24-metrics] нет чекпоинта ${run} — пропуск" >&2; continue; fi
  echo "[exp24-metrics] ${run} ckpt=${ckpt}"
  python "${EVAL}" \
    --checkpoint "${ckpt}" \
    --memmap "${MEMMAP}" \
    --climatology "${CLIMATOLOGY}" \
    --thresholds "${THRESHOLDS}" \
    --out "${OUT_DIR}/metrics_${run}.npz" \
    --out-per-sample "${RAW_DIR}/metrics_${run}_per_sample.npz" \
    --batch-size 8 \
    --num-workers 8
done

# Парные дельты к nophys-orog — только если контроль и ≥1 другой арм посчитаны.
baseline="exp24-iam4vp-nophys-orog-${REGION}"
if ls "${RAW_DIR}/metrics_${baseline}-s0_per_sample.npz" >/dev/null 2>&1 \
   && [[ "$(ls "${RAW_DIR}"/metrics_*_per_sample.npz 2>/dev/null | wc -l)" -ge 2 ]]; then
  echo "[exp24-metrics] парные дельты к ${baseline}"
  python "${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/paired_deltas.py" \
    --per-sample-dir "${RAW_DIR}" \
    --baseline "${baseline}" \
    --out "${OUT_DIR}/paired_deltas_iam4vp_${REGION}.npz"
else
  echo "[exp24-metrics] пропуск paired_deltas: нужен baseline + ≥1 арм в ${RAW_DIR}"
fi
echo "[exp24-metrics] готово -> ${OUT_DIR}"
