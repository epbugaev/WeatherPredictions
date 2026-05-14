#!/bin/bash
#SBATCH --job-name=check-physics-predformergft
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# 72h физическая проверка метода PredFormerGFT (WENO-5 + Euler + beta-plane).
# CPU-only.
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/check_physics_predformergft.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

MEMMAP_PATH="${MEMMAP_PATH:-/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat}"
MEAN_STD_PATH="${MEAN_STD_PATH:-/home/epbugaev/weather_bench/1.40625deg/mean_std.npy}"
HORIZON="${HORIZON_HOURS:-72}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

echo "[check_physics_predformergft] JobID=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[check_physics_predformergft] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[check_physics_predformergft] memmap=${MEMMAP_PATH}, mean_std=${MEAN_STD_PATH}"
echo "[check_physics_predformergft] horizon=${HORIZON}h, block_dt=${BLOCK_DT}s, year=${YEAR}"

python tools/check_physics_predformergft.py \
  --memmap-path "${MEMMAP_PATH}" \
  --mean-std-path "${MEAN_STD_PATH}" \
  --horizon-hours "${HORIZON}" \
  --block-dt-seconds "${BLOCK_DT}" \
  --year "${YEAR}" \
  "$@"
