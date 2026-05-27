#!/bin/bash
#SBATCH --job-name=PredFormer-USA-2gpu
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-18:00:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Production training of PredFormer-USA on 2 GPUs (single-node DDP) with the
# §2.5 memmap path:
#   1. Stage /home/fa.buzaev/era5_memmap/predformer_usa_2000_2004.dat to
#      /tmp on the compute node (1-2 min; ~25 GB on InfiniBand).
#   2. Export MEMMAP_PATH_OVERRIDE so train.build_dataset reads the staged
#      copy without rewriting configs/predformer_usa_memmap.yaml.
#   3. Run torchrun --nproc_per_node=2 via the canonical launcher.
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/fa.buzaev/WeatherPredictions}}"
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"
if [[ ! -f "${weatherpred__launcher}" ]]; then
  weatherpred__launcher="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/launch_train.sh"
fi

# Memmap staging — same trick as bench step4 (which reduced data_wait by
# ~2400x vs Lustre random access).
ORIG_MEMMAP="${ORIG_MEMMAP:-/home/fa.buzaev/era5_memmap/predformer_usa_2000_2004.dat}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[train-2gpu] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
weatherpred__stage_start=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
weatherpred__stage_secs=$(( $(date +%s) - weatherpred__stage_start ))
echo "[train-2gpu] staged in ${weatherpred__stage_secs}s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"

trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

# Two GPUs per node; launch_train.sh reads NGPUS from SLURM_GPUS_ON_NODE.
exec bash "${weatherpred__launcher}" predformer_usa_memmap "$@"
