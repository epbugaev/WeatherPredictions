#!/bin/bash
#SBATCH --job-name=sanity-pi-iam4vp-1gpu
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Deep single-GPU sanity diagnostics for all PI-IAM4VP v4 arms on real data.
# Stages the packed USA memmap to node-local storage (same flow as
# train_pi_iam4vp_residual_usa_v4_2gpu.sh) and runs tools/sanity_pi_iam4vp_gpu.py.
# Extra positional args are forwarded to the python tool (e.g. --arms fixedeq).
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[sanity-1gpu] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"

trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

exec python tools/sanity_pi_iam4vp_gpu.py \
  --out "logs/sanity_pi_iam4vp_${SLURM_JOB_ID:-local}.json" "$@"
