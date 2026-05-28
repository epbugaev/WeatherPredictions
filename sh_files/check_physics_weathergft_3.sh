#!/bin/bash
#SBATCH --job-name=phys-fd4-rk4-sphere-radmix
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# 72h физическая проверка метода WeatherGFT_3 (FD-4 + RK4 + spherical Coriolis +
# turbulent mixing + radiative cooling + convective precipitation).
# CPU-only. ~4×Euler из-за RK4.
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/check_physics_weathergft_3.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

MEMMAP_PATH="${MEMMAP_PATH:-/home/ebugaev/era5_memmap/predformer_globe_2000_2018.dat}"
MEAN_STD_PATH="${MEAN_STD_PATH:-}"
HORIZON="${HORIZON_HOURS:-48}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

echo "[check_physics_weathergft_3] JobID=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[check_physics_weathergft_3] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[check_physics_weathergft_3] memmap=${MEMMAP_PATH}, mean_std=${MEAN_STD_PATH}"
echo "[check_physics_weathergft_3] horizon=${HORIZON}h, block_dt=${BLOCK_DT}s, year=${YEAR}"

python tools/check_physics_weathergft_3.py \
  --memmap-path "${MEMMAP_PATH}" \
  --mean-std-path "${MEAN_STD_PATH}" \
  --horizon-hours "${HORIZON}" \
  --block-dt-seconds "${BLOCK_DT}" \
  --year "${YEAR}" \
  "$@"
