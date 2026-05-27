#!/bin/bash
#SBATCH --job-name=SimVP-USA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-18:00:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# SimVP на USA-датасете.
# Submit: sbatch sh_files/train_SimVp_USA.sh
# Конфиг:  configs/simvp_usa.yaml
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

exec bash "${weatherpred__launcher}" simvp_usa "$@"
