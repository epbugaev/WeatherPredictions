#!/bin/bash
#SBATCH --job-name=physics-stats-eq15
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Эксперимент 15: невязки всех пяти уравнений на ERA5 — матрица вариантов
# (база exp 13/14 + диабатика, источники q, диагностический z, ω-варианты).
# Параметры — ПОЗИЦИОННЫЕ аргументы sbatch (remote_submit не пробрасывает env
# через ssh): $1 — python-скрипт относительно REPO_ROOT, $2 — YEAR, $3 — DOMAIN
# (usa|globe), $4 — STRIDE, $5 — MAX_TRIPLES (0 = все), $6 — суффикс OUT.
# Пишет JSON в logs/.
set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# sbatch копирует скрипт в spool-каталог: REPO_ROOT восстанавливаем из submit dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

SCRIPT_REL="${1:?usage: sbatch physics_stats_eq15_cpu.sh <script.py> <year> <domain> [stride] [max_triples] [tag] [clim_out|-] [clim_in|-]}"
export YEAR="${2:?year}"
export DOMAIN="${3:?domain}"
export STRIDE="${4:-1}"
export MAX_TRIPLES="${5:-0}"
TAG="${6:-run}"
CLIM_OUT_ARG="${7:--}"
CLIM_IN_ARG="${8:--}"

JOB="${SLURM_JOB_ID:-local}"
OUT="${REPO_ROOT}/logs/eq15_${TAG}_${DOMAIN}_${YEAR}_${JOB}.json"
export OUT
export MAPS_OUT="${REPO_ROOT}/logs/eq15_${TAG}_maps_${DOMAIN}_${YEAR}_${JOB}.npz"
if [[ "${CLIM_OUT_ARG}" != "-" ]]; then
  export CLIM_OUT="${CLIM_OUT_ARG}"
fi
if [[ "${CLIM_IN_ARG}" != "-" ]]; then
  export CLIM_IN="${CLIM_IN_ARG}"
fi

echo "[eq15] script=${SCRIPT_REL} year=${YEAR} domain=${DOMAIN} stride=${STRIDE} max_triples=${MAX_TRIPLES} out=${OUT}"
python "${REPO_ROOT}/${SCRIPT_REL}"
echo "[eq15] done: ${OUT}"
