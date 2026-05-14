#!/bin/bash
#SBATCH --job-name=check-physics-weathergft
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# 72h физическая проверка метода WeatherGFT (FD-4 + Euler + const Coriolis).
# CPU-only. Никаких GPU (--gres не указан).
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/check_physics_weathergft.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/fa.buzaev/WeatherPredictions}}"
cd "${REPO_ROOT}"

# Comet creds через .env
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MEMMAP_PATH="${MEMMAP_PATH:-/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat}"
MEAN_STD_PATH="${MEAN_STD_PATH:-/home/epbugaev/weather_bench/1.40625deg/mean_std.npy}"
HORIZON="${HORIZON_HOURS:-72}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

echo "[check_physics_weathergft] JobID=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[check_physics_weathergft] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[check_physics_weathergft] memmap=${MEMMAP_PATH}, mean_std=${MEAN_STD_PATH}"
echo "[check_physics_weathergft] horizon=${HORIZON}h, block_dt=${BLOCK_DT}s, year=${YEAR}"

python tools/check_physics_weathergft.py \
  --memmap-path "${MEMMAP_PATH}" \
  --mean-std-path "${MEAN_STD_PATH}" \
  --horizon-hours "${HORIZON}" \
  --block-dt-seconds "${BLOCK_DT}" \
  --year "${YEAR}" \
  "$@"
