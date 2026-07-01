#!/bin/bash
#SBATCH --job-name=pde-matrix-usa-v4
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Train a tiny matrix calibrator over PurePDEKernel tendencies on v4 USA memmap.
# Usage:
#   sbatch sh_files/train_pde_matrix_operator_usa_v4.sh pi_iam4vp_residual_usa_v4
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
  echo "[pde-matrix] Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_pde_matrix_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[pde-matrix] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__t0=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
echo "[pde-matrix] staged in $(( $(date +%s) - weatherpred__t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"
trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
IDENTITY_LAMBDA="${IDENTITY_LAMBDA:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MATRIX_MODE="${MATRIX_MODE:-per_level}"
MAX_DELTA="${MAX_DELTA:-0.5}"
Q_SCALE="${Q_SCALE:-0.01}"
SUBSTEPS_PER_FRAME="${SUBSTEPS_PER_FRAME:-12}"
BLOCK_DT="${BLOCK_DT:-300}"
STENCIL="${STENCIL:-fd4}"
CORIOLIS="${CORIOLIS:-spherical}"
TIME_SCHEME="${TIME_SCHEME:-euler}"
LAT_RANGE_LOW="${LAT_RANGE_LOW:-24}"
LAT_RANGE_HIGH="${LAT_RANGE_HIGH:-56}"
HUMIDITY_MODE="${HUMIDITY_MODE:-relative_to_specific}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/pde_matrix_operator/${SLURM_JOB_ID:-local}_${MATRIX_MODE}}"

extra_args=()
if [[ -n "${HORIZON:-}" ]]; then
  extra_args+=(--horizon "${HORIZON}")
fi
if [[ -n "${TRAIN_MAX_BATCHES:-}" ]]; then
  extra_args+=(--train-max-batches "${TRAIN_MAX_BATCHES}")
fi
if [[ -n "${VAL_MAX_BATCHES:-}" ]]; then
  extra_args+=(--val-max-batches "${VAL_MAX_BATCHES}")
fi
if [[ "${USE_UNIVERSAL_R:-0}" == "1" ]]; then
  extra_args+=(--use-universal-R)
fi

echo "[pde-matrix] config=${CONFIG_PATH}"
echo "[pde-matrix] output_dir=${OUTPUT_DIR}"
echo "[pde-matrix] python=$(command -v python)"
echo "[pde-matrix] matrix=${MATRIX_MODE}, epochs=${EPOCHS}, lr=${LR}, horizon=${HORIZON:-config}"
echo "[pde-matrix] kernel=${STENCIL}/${CORIOLIS}/${TIME_SCHEME}, block_dt=${BLOCK_DT}, substeps=${SUBSTEPS_PER_FRAME}"

exec python tools/train_pde_matrix_operator.py \
  --config "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --identity-lambda "${IDENTITY_LAMBDA}" \
  --grad-clip "${GRAD_CLIP}" \
  --matrix-mode "${MATRIX_MODE}" \
  --max-delta "${MAX_DELTA}" \
  --q-scale "${Q_SCALE}" \
  --substeps-per-frame "${SUBSTEPS_PER_FRAME}" \
  --block-dt "${BLOCK_DT}" \
  --stencil "${STENCIL}" \
  --coriolis "${CORIOLIS}" \
  --time-scheme "${TIME_SCHEME}" \
  --lat-range-deg "${LAT_RANGE_LOW}" "${LAT_RANGE_HIGH}" \
  --humidity-mode "${HUMIDITY_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"
