#!/bin/bash
#SBATCH --job-name=train-old-region-v4-2gpu
#SBATCH --partition=rocky
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Generic v4 old-region launcher. Pass a config stem, e.g.
#   sbatch sh_files/train_old_region_v4_2gpu.sh simvp_old_region_v4_sanity
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

CONFIG_STEM="${1:?Usage: sbatch sh_files/train_old_region_v4_2gpu.sh <config_stem>}"
shift

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
weatherpred__env_file="${weatherpred__repo_root}/.env"
if [[ -f "${weatherpred__env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${weatherpred__env_file}"
  set +a
  weatherpred__repo_root="${REPO_ROOT:-${weatherpred__repo_root}}"
fi
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"
if [[ ! -f "${weatherpred__launcher}" ]]; then
  weatherpred__launcher="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/launch_train.sh"
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_OLD_REGION_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_OLD_REGION_MEMMAP to the packed old-region memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[train-old-region-v4] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__t0=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
echo "[train-old-region-v4] staged in $(( $(date +%s) - weatherpred__t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"

trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

exec bash "${weatherpred__launcher}" "${CONFIG_STEM}" "$@"
