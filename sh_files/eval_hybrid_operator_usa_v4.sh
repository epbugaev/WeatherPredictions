#!/bin/bash
#SBATCH --job-name=eval-hybrid-operator-usa-v4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Evaluate HybridBlock-only operator vs persistence on v4 USA memmap.
# Usage:
#   sbatch sh_files/eval_hybrid_operator_usa_v4.sh /path/to/best.pt pi_iam4vp_residual_usa_v4
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 <checkpoint.pt> [config_stem|config_path]" >&2
  echo "Example: $0 /home/ebugaev/checkpoints/.../best.pt pi_iam4vp_residual_usa_v4" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG_ARG="${1:-${CONFIG:-pi_iam4vp_residual_usa_v4}}"
if [[ "${CONFIG_ARG}" == *.yaml || "${CONFIG_ARG}" == */* ]]; then
  CONFIG_PATH="${CONFIG_ARG}"
else
  CONFIG_PATH="configs/${CONFIG_ARG}.yaml"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[eval-hybrid] Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_eval_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[eval-hybrid] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__t0=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
echo "[eval-hybrid] staged in $(( $(date +%s) - weatherpred__t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"
trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

OUTPUT_CSV="${OUTPUT_CSV:-logs/hybrid_operator_${SLURM_JOB_ID:-local}.csv}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

extra_args=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  extra_args+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${HORIZON:-}" ]]; then
  extra_args+=(--horizon "${HORIZON}")
fi
if [[ -n "${HYBRID_MODE:-}" ]]; then
  extra_args+=(--hybrid-mode "${HYBRID_MODE}")
fi
if [[ "${AUTOREGRESSIVE:-1}" == "0" ]]; then
  extra_args+=(--no-autoregressive)
fi
if [[ "${STRICT_CHECKPOINT:-1}" == "0" ]]; then
  extra_args+=(--no-strict)
fi

echo "[eval-hybrid] config=${CONFIG_PATH}"
echo "[eval-hybrid] checkpoint=${CHECKPOINT}"
echo "[eval-hybrid] output_csv=${OUTPUT_CSV}"
echo "[eval-hybrid] python=$(command -v python)"

exec python tools/evaluate_hybrid_operator.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --output-csv "${OUTPUT_CSV}" \
  "${extra_args[@]}"
