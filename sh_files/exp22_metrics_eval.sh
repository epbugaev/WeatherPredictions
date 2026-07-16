#!/bin/bash
#SBATCH --job-name=exp22-metrics
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик армов одного (семейство × регион) exp22 с общей эпохи 150 +
# парные бутстрап-дельты к арму no_physics. Диспатчит на per-family metrics_eval
# (SimVPv2→exp21, PredRNNv2→exp20, IAM4VP→exp16 — носитель прогноза у каждого свой),
# ядро метрик и схема npz общие. Регион задаёт мемап/климатологию/пороги.
#
#   EXP22_FAMILY=simvpv2 EXP22_REGION=npac sbatch sh_files/exp22_metrics_eval.sh
#   EXP22_FAMILY=simvpv2 EXP22_REGION=france EXP22_ARMS="nophys" sbatch ...   # подмножество
set -euo pipefail
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp21}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

FAMILY="${EXP22_FAMILY:?Set EXP22_FAMILY=simvpv2|predrnnv2|iam4vp}"
REGION="${EXP22_REGION:?Set EXP22_REGION=france|npac}"
CKPT_BASE="${CKPT_BASE:-${REPO_ROOT}/checkpoints}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
OUT_DIR="${OUT_DIR:-${HOME}/exp22_metrics}"
# Raw по-сэмпльно ИЗОЛИРОВАН по (семейство,регион): paired_deltas.py глобит весь
# каталог, поэтому в нём должны лежать только 3 арма одной лестницы, не вперемешку.
RAW_DIR="${RAW_DIR:-${HOME}/exp22_metrics_raw/${FAMILY}_${REGION}}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

case "${FAMILY}" in
  simvpv2) EVAL="${REPO_ROOT}/docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_eval.py" ;;
  predrnnv2) EVAL="${REPO_ROOT}/docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_eval.py" ;;
  # IAM4VP exp22 — 6-native (time_prediction=6); нативный горизонт, без 2x-раскатки
  # (exp16-обёртка катит до 12 и падает на mask_token размера 6).
  iam4vp) EVAL="${REPO_ROOT}/docs/experiments/22_cross_region_physics/metrics/metrics_eval_iam4vp.py" ;;
  *) echo "[exp22-metrics] неизвестное семейство ${FAMILY}" >&2; exit 2 ;;
esac

MEMMAP="${MEMMAP_DIR}/predformer_${REGION}_2000_2004.dat"
CLIMATOLOGY="${MEMMAP_DIR}/climatology_${REGION}_2000_2003.npz"
THRESHOLDS="${MEMMAP_DIR}/thresholds_${REGION}_2004.npz"
for f in "${MEMMAP}" "${CLIMATOLOGY}" "${THRESHOLDS}"; do
  if [[ ! -f "${f}" ]]; then echo "[exp22-metrics] нет ${f}" >&2; exit 2; fi
done

# Порядок: контроль no_physics первым (он же baseline парного бутстрапа).
read -r -a ARMS <<<"${EXP22_ARMS:-nophys a2 legacy}"
for arm in "${ARMS[@]}"; do
  run="exp22-${FAMILY}-${arm}-${REGION}-s0"
  ckpt="$(ls -t "${CKPT_BASE}/${run}"/*/"${CKPT_NAME}" 2>/dev/null | head -1)"
  if [[ -z "${ckpt}" ]]; then echo "[exp22-metrics] нет чекпоинта ${run} — пропуск" >&2; continue; fi
  echo "[exp22-metrics] ${run} ckpt=${ckpt}"
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

# Парные дельты к no_physics — только если контроль и хотя бы один физ-арм посчитаны
# (RAW_DIR изолирован по семейству/региону, поэтому глоб paired_deltas корректен).
baseline="exp22-${FAMILY}-nophys-${REGION}"
if ls "${RAW_DIR}/metrics_${baseline}-s0_per_sample.npz" >/dev/null 2>&1 \
   && [[ "$(ls "${RAW_DIR}"/metrics_*_per_sample.npz 2>/dev/null | wc -l)" -ge 2 ]]; then
  echo "[exp22-metrics] парные дельты к ${baseline}"
  python "${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/paired_deltas.py" \
    --per-sample-dir "${RAW_DIR}" \
    --baseline "${baseline}" \
    --out "${OUT_DIR}/paired_deltas_${FAMILY}_${REGION}.npz"
else
  echo "[exp22-metrics] пропуск paired_deltas: нужен baseline + ≥1 физ-арм в ${RAW_DIR}"
fi
echo "[exp22-metrics] готово -> ${OUT_DIR}"
