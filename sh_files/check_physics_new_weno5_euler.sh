#!/bin/bash
#SBATCH --job-name=phys-new-weno5-euler
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# 72h физическая проверка НОВОЙ семьи (utils.physics.PurePDEKernel):
# WENO-5 + Forward Euler + spherical Coriolis + reflect h-boundary.
# CPU-only.
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

: "${MEMMAP_PATH:?Set MEMMAP_PATH to a packed globe memmap .dat before running this checker.}"
MEAN_STD_PATH="${MEAN_STD_PATH:-}"
HORIZON="${HORIZON_HOURS:-48}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

echo "[phys-new-weno5-euler] JobID=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[phys-new-weno5-euler] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python tools/check_physics_new_kernel.py \
  --stencil weno5 \
  --time-scheme euler \
  --coriolis spherical \
  --boundary-x reflect \
  --boundary-y replicate \
  --boundary-z replicate \
  --memmap-path "${MEMMAP_PATH}" \
  --mean-std-path "${MEAN_STD_PATH}" \
  --horizon-hours "${HORIZON}" \
  --block-dt-seconds "${BLOCK_DT}" \
  --year "${YEAR}" \
  "$@"
