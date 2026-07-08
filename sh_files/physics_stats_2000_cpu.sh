#!/bin/bash
#SBATCH --job-name=physics-stats-2000
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Пост-фикс статистика физики (аудит 13) на реальном ERA5 за 2000 год:
# PDE-невязки / ω-тест / гидростатика / конденсация / CFL, USA-кроп + глобус.
# Читает netCDF из /home/fratnikov/weather_bench/1.40625deg/, пишет JSON в logs/.
set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# sbatch копирует скрипт в spool-каталог: REPO_ROOT восстанавливаем из submit dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

JOB="${SLURM_JOB_ID:-local}"
OUT="${OUT:-${REPO_ROOT}/logs/physics_stats_2000_${JOB}.json}"
export OUT
export N_PAIRS="${N_PAIRS:-96}"
export YEAR="${YEAR:-2000}"
export MAPS_OUT="${MAPS_OUT:-${REPO_ROOT}/logs/physics_maps_2000_${JOB}.npz}"

echo "[physics-stats] year=${YEAR} n_pairs=${N_PAIRS} out=${OUT}"
python "${REPO_ROOT}/docs/experiments/13_sign_convention_fix/physics_stats_2000.py"
echo "[physics-stats] done: ${OUT}"
