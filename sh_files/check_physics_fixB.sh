#!/bin/bash
#SBATCH --job-name=phys-fixB-PaUnits-Rd
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Fix B: оставить broken Q, но pressure в Pa + R_d в гидростатике.
# Ожидается, что blow-up сохранится (см. docstring fix.py).
# CPU-only.
# =============================================================================
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1
REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

: "${MEMMAP_PATH:?Set MEMMAP_PATH to a packed globe memmap .dat before running this checker.}"
HORIZON="${HORIZON_HOURS:-48}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

echo "[phys-fixB] JobID=${SLURM_JOB_ID:-local} host=$(hostname) git_sha=$(git rev-parse --short HEAD 2>/dev/null)"
python tools/check_physics_fix.py \
  --fix-mode B \
  --memmap-path "${MEMMAP_PATH}" \
  --horizon-hours "${HORIZON}" \
  --block-dt-seconds "${BLOCK_DT}" \
  --year "${YEAR}" \
  "$@"
