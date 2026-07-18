#!/bin/bash
#SBATCH --job-name=exp25-metrics
#SBATCH --account=proj_1717
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик 4 армов exp25 (изоляция маршрута орографии, USA, 12-шаг
# exp16-обёртка) + парные бутстрап-дельты к baseline d0. Носитель — IAM4VP.
#
# Чекпоинты: d0/d2 обучены в exp25, d1/d3 переиспользуются (см. README):
#   d0,d2 -> ${CKPT_BASE}/exp25-iam4vp-<arm>-usa-s0       (обучены в exp25)
#   d1    -> abl16_long_ckpt/abl16L-r3-a2-exp13-t12-s0     (готов @500, идентичный конфиг)
#   d3    -> wt_exp24/.../exp24-iam4vp-a2-orog-usa-s1      (готов @500 в exp24, seed 1)
# Все выходы именуются exp25-iam4vp-<arm>-usa-s0 → единое каноническое имя.
#
#   sbatch sh_files/exp25_metrics_eval.sh
#   EXP25_ARMS="d0 d1" sbatch sh_files/exp25_metrics_eval.sh   # подмножество
set -euo pipefail
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp25}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

REGION="usa"
CKPT_BASE="${CKPT_BASE:-${REPO_ROOT}/checkpoints}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
OUT_DIR="${OUT_DIR:-${HOME}/exp25_metrics}"
RAW_DIR="${RAW_DIR:-${HOME}/exp25_metrics_raw/iam4vp_${REGION}}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

# Родитель армов — abl16 t12: 12-шаговый rollout, eval-обёртка exp16.
EVAL="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/metrics_eval.py"

MEMMAP="${MEMMAP_DIR}/predformer_${REGION}_2000_2004.dat"
CLIMATOLOGY="${MEMMAP_DIR}/climatology_${REGION}_2000_2003.npz"
THRESHOLDS="${MEMMAP_DIR}/thresholds_${REGION}_2004.npz"
for f in "${MEMMAP}" "${CLIMATOLOGY}" "${THRESHOLDS}"; do
  if [[ ! -f "${f}" ]]; then echo "[exp25-metrics] нет ${f}" >&2; exit 2; fi
done

# Резолвер каталога чекпоинта арма (d1/d3 переиспользуются из готовых ранов).
arm_ckpt_dir() {
  case "$1" in
    d1) echo "${HOME}/abl16_long_ckpt/abl16L-r3-a2-exp13-t12-s0" ;;
    d3) echo "${HOME}/wt_exp24/checkpoints/exp24-iam4vp-a2-orog-usa-s1" ;;
    *)  echo "${CKPT_BASE}/exp25-iam4vp-$1-usa-s0" ;;
  esac
}

# Порядок: baseline d0 первым (он же контроль парного бутстрапа).
read -r -a ARMS <<<"${EXP25_ARMS:-d0 d1 d2 d3}"
for arm in "${ARMS[@]}"; do
  run="exp25-iam4vp-${arm}-${REGION}-s0"
  dir="$(arm_ckpt_dir "${arm}")"
  ckpt="$(ls -t "${dir}"/*/"${CKPT_NAME}" 2>/dev/null | head -1)"
  if [[ -z "${ckpt}" ]]; then echo "[exp25-metrics] нет чекпоинта ${arm} в ${dir} — пропуск" >&2; continue; fi
  echo "[exp25-metrics] ${run} ckpt=${ckpt}"
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

# Парные дельты к d0 — только если контроль и ≥1 другой арм посчитаны.
baseline="exp25-iam4vp-d0-${REGION}"
if ls "${RAW_DIR}/metrics_${baseline}-s0_per_sample.npz" >/dev/null 2>&1 \
   && [[ "$(ls "${RAW_DIR}"/metrics_*_per_sample.npz 2>/dev/null | wc -l)" -ge 2 ]]; then
  echo "[exp25-metrics] парные дельты к ${baseline}"
  python "${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/paired_deltas.py" \
    --per-sample-dir "${RAW_DIR}" \
    --baseline "${baseline}" \
    --out "${OUT_DIR}/paired_deltas_iam4vp_${REGION}.npz"
else
  echo "[exp25-metrics] пропуск paired_deltas: нужен baseline d0 + ≥1 арм в ${RAW_DIR}"
fi
echo "[exp25-metrics] готово -> ${OUT_DIR}"
