#!/bin/bash
#SBATCH --job-name=exp26-metrics
#SBATCH --account=proj_1894
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Полный набор метрик армов одного (семейство × регион) exp26 (кросс-региональная
# реплика exp24 — статические входы) с общей эпохи 499/39 + парные бутстрап-дельты
# к арму no_physics+static. Диспатчит на per-family metrics_eval (SimVPv2→exp21,
# PredRNNv2→exp20, IAM4VP→exp16 — носитель прогноза у каждого свой), ядро метрик и
# схема npz общие (см. sh_files/exp22_metrics_eval.sh — тот же паттерн). Регион
# задаёт мемап/климатологию/пороги.
#
# IAM4VP несёт полный A/B (orog/static) — 4 арма; SimVPv2/PredRNNv2 — только
# static — 2 арма (как в exp24/exp26 матрице).
#
#   EXP26_FAMILY=iam4vp EXP26_REGION=france sbatch sh_files/exp26_metrics_eval.sh
#   EXP26_FAMILY=simvpv2 EXP26_REGION=npac EXP26_ARMS="nophys-static" sbatch ...
set -euo pipefail
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp26}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

FAMILY="${EXP26_FAMILY:?Set EXP26_FAMILY=simvpv2|predrnnv2|iam4vp}"
REGION="${EXP26_REGION:?Set EXP26_REGION=france|npac}"
CKPT_BASE="${CKPT_BASE:-${REPO_ROOT}/checkpoints}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
OUT_DIR="${OUT_DIR:-${HOME}/exp26_metrics}"
# Raw по-сэмпльно ИЗОЛИРОВАН по (семейство,регион): paired_deltas.py глобит весь
# каталог, поэтому в нём должны лежать только армы одной лестницы, не вперемешку.
RAW_DIR="${RAW_DIR:-${HOME}/exp26_metrics_raw/${FAMILY}_${REGION}}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
mkdir -p "${OUT_DIR}" "${RAW_DIR}"

case "${FAMILY}" in
  simvpv2) EVAL="${REPO_ROOT}/docs/experiments/21_pi_simvpv2_ladder/metrics/metrics_eval.py" ;;
  predrnnv2) EVAL="${REPO_ROOT}/docs/experiments/20_pi_predrnnv2_ladder/metrics/metrics_eval.py" ;;
  # exp26 IAM4VP конфиги наследуют abl16_long t12 (T_data=12) — родная 12-шаговая
  # обёртка exp16, как у exp24 (не exp22-шный 6-native вариант).
  iam4vp) EVAL="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/metrics_eval.py" ;;
  *) echo "[exp26-metrics] неизвестное семейство ${FAMILY}" >&2; exit 2 ;;
esac

MEMMAP="${MEMMAP_DIR}/predformer_${REGION}_2000_2004.dat"
CLIMATOLOGY="${MEMMAP_DIR}/climatology_${REGION}_2000_2003.npz"
THRESHOLDS="${MEMMAP_DIR}/thresholds_${REGION}_2004.npz"
for f in "${MEMMAP}" "${CLIMATOLOGY}" "${THRESHOLDS}"; do
  if [[ ! -f "${f}" ]]; then echo "[exp26-metrics] нет ${f}" >&2; exit 2; fi
done

# Порядок: контроль nophys-static первым (он же baseline парного бутстрапа).
# IAM4VP несёт ещё orog-пару (тест H1: достаточна ли одна орография).
case "${FAMILY}" in
  iam4vp) DEFAULT_ARMS="nophys-static a2-static nophys-orog a2-orog" ;;
  *) DEFAULT_ARMS="nophys-static a2-static" ;;
esac
read -r -a ARMS <<<"${EXP26_ARMS:-${DEFAULT_ARMS}}"
for arm in "${ARMS[@]}"; do
  run="exp26-${FAMILY}-${arm}-${REGION}-s0"
  ckpt="$(ls -t "${CKPT_BASE}/${run}"/*/"${CKPT_NAME}" 2>/dev/null | head -1)"
  if [[ -z "${ckpt}" ]]; then echo "[exp26-metrics] нет чекпоинта ${run} — пропуск" >&2; continue; fi
  echo "[exp26-metrics] ${run} ckpt=${ckpt}"
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

# Парные дельты к nophys-static — только если контроль и ≥1 другой арм посчитаны
# (RAW_DIR изолирован по семейству/региону, поэтому глоб paired_deltas корректен).
baseline="exp26-${FAMILY}-nophys-static-${REGION}"
if ls "${RAW_DIR}/metrics_${baseline}-s0_per_sample.npz" >/dev/null 2>&1 \
   && [[ "$(ls "${RAW_DIR}"/metrics_*_per_sample.npz 2>/dev/null | wc -l)" -ge 2 ]]; then
  echo "[exp26-metrics] парные дельты к ${baseline}"
  python "${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics/paired_deltas.py" \
    --per-sample-dir "${RAW_DIR}" \
    --baseline "${baseline}" \
    --out "${OUT_DIR}/paired_deltas_${FAMILY}_${REGION}.npz"
else
  echo "[exp26-metrics] пропуск paired_deltas: нужен baseline + ≥1 арм в ${RAW_DIR}"
fi
echo "[exp26-metrics] готово -> ${OUT_DIR}"
