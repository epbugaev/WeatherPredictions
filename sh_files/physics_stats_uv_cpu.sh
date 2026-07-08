#!/bin/bash
#SBATCH --job-name=physics-stats-uv
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Эксперимент 14: невязки уравнений движения (u, v) на ERA5 — матрица
# вариантов (d_y-ориентация, точные широты, кривизна, d_z, трение, формы
# адвекции/ω) + разложение по членам/уровням/широтам. Параметры через env:
# YEAR (2000), DOMAIN (usa|globe), STRIDE (часов между тройками снимков),
# MAX_TRIPLES (0 = все). Пишет JSON в logs/.
set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# sbatch копирует скрипт в spool-каталог: REPO_ROOT восстанавливаем из submit dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

JOB="${SLURM_JOB_ID:-local}"
export YEAR="${YEAR:-2000}"
export DOMAIN="${DOMAIN:-globe}"
export STRIDE="${STRIDE:-4}"
export MAX_TRIPLES="${MAX_TRIPLES:-0}"
OUT="${OUT:-${REPO_ROOT}/logs/physics_stats_uv_${DOMAIN}_${YEAR}_${JOB}.json}"
export OUT

echo "[physics-stats-uv] year=${YEAR} domain=${DOMAIN} stride=${STRIDE} max_triples=${MAX_TRIPLES} out=${OUT}"
python "${REPO_ROOT}/docs/experiments/14_uv_residual_improvement/physics_stats_uv.py"
echo "[physics-stats-uv] done: ${OUT}"
