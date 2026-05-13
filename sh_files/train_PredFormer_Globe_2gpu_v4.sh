#!/bin/bash
#SBATCH --job-name=PredFormer-Globe-2gpu-v4
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=6-00:00:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Production training of PredFormer-Globe-v4 on 2 GPUs.
# Same recipe as ``train_PredFormer_USA_2gpu_v4.sh`` (bf16 + fused AdamW +
# pin_memory + persistent_workers=false + grad_clip + warmup) — the only
# differences are the config (full-globe 32x64 grid, 2000-2018 train range)
# and the memmap path (predformer_globe_2000_2018.dat, ~95 GB).
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/fa.buzaev/WeatherPredictions}}"
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"
if [[ ! -f "${weatherpred__launcher}" ]]; then
  weatherpred__launcher="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/launch_train.sh"
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[train-globe-2gpu-v4] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__stage_start=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
weatherpred__stage_secs=$(( $(date +%s) - weatherpred__stage_start ))
echo "[train-globe-2gpu-v4] staged in ${weatherpred__stage_secs}s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"

trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

exec bash "${weatherpred__launcher}" predformer_globe_v4 "$@"
