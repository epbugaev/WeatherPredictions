#!/bin/bash
#SBATCH --job-name=eval-pure-pde-usa-v4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Evaluate equations-only PurePDEKernel vs persistence on v4 USA memmap.
# Usage:
#   sbatch sh_files/eval_pure_pde_operator_usa_v4.sh pi_iam4vp_residual_usa_v4
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

CONFIG_ARG="${1:-${CONFIG:-pi_iam4vp_residual_usa_v4}}"
if [[ "${CONFIG_ARG}" == *.yaml || "${CONFIG_ARG}" == */* ]]; then
  CONFIG_PATH="${CONFIG_ARG}"
else
  CONFIG_PATH="configs/${CONFIG_ARG}.yaml"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[eval-pure-pde] Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_pure_pde_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[eval-pure-pde] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__t0=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
echo "[eval-pure-pde] staged in $(( $(date +%s) - weatherpred__t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"
trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

OUTPUT_CSV="${OUTPUT_CSV:-logs/pure_pde_operator_${SLURM_JOB_ID:-local}.csv}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
DEVICE="${DEVICE:-cuda}"
LAT_RANGE_LOW="${LAT_RANGE_LOW:-24}"
LAT_RANGE_HIGH="${LAT_RANGE_HIGH:-56}"
SUBSTEPS_PER_FRAME="${SUBSTEPS_PER_FRAME:-12}"
BLOCK_DT="${BLOCK_DT:-300}"
STENCIL="${STENCIL:-fd4}"
CORIOLIS="${CORIOLIS:-spherical}"
TIME_SCHEME="${TIME_SCHEME:-euler}"
HUMIDITY_MODE="${HUMIDITY_MODE:-relative_to_specific}"

extra_args=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  extra_args+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${HORIZON:-}" ]]; then
  extra_args+=(--horizon "${HORIZON}")
fi
if [[ "${USE_UNIVERSAL_R:-0}" == "1" ]]; then
  extra_args+=(--use-universal-R)
fi

echo "[eval-pure-pde] config=${CONFIG_PATH}"
echo "[eval-pure-pde] output_csv=${OUTPUT_CSV}"
echo "[eval-pure-pde] python=$(command -v python)"
echo "[eval-pure-pde] kernel=${STENCIL}/${CORIOLIS}/${TIME_SCHEME}, block_dt=${BLOCK_DT}, substeps=${SUBSTEPS_PER_FRAME}"

exec python tools/evaluate_pure_pde_operator.py \
  --config "${CONFIG_PATH}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --substeps-per-frame "${SUBSTEPS_PER_FRAME}" \
  --block-dt "${BLOCK_DT}" \
  --stencil "${STENCIL}" \
  --coriolis "${CORIOLIS}" \
  --time-scheme "${TIME_SCHEME}" \
  --lat-range-deg "${LAT_RANGE_LOW}" "${LAT_RANGE_HIGH}" \
  --humidity-mode "${HUMIDITY_MODE}" \
  --output-csv "${OUTPUT_CSV}" \
  "${extra_args[@]}"
