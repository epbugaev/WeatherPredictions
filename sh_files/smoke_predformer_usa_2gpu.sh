#!/bin/bash
#SBATCH --job-name=smoke-predformer-usa-2gpu
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/fa.buzaev/WeatherPredictions}}"
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"

# Memmap staging — same as production train_PredFormer_USA_2gpu.sh.
ORIG_MEMMAP="${ORIG_MEMMAP:-/home/fa.buzaev/era5_memmap/predformer_usa_2000_2004.dat}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[smoke-predformer] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"
trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

exec bash "${weatherpred__launcher}" predformer_usa_memmap "$@"
