#!/bin/bash
#SBATCH --job-name=train-usa-2gpu
#SBATCH --partition=rocky
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=6-00:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Generic USA production launcher for legacy WeatherBench files.
# Usage: sbatch sh_files/train_usa_2gpu.sh simvp_usa
set -euo pipefail

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"
if [[ ! -f "${weatherpred__launcher}" ]]; then
  weatherpred__launcher="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/launch_train.sh"
fi

CONFIG_STEM="${1:?Usage: sbatch sh_files/train_usa_2gpu.sh <config_stem>}"
shift

exec bash "${weatherpred__launcher}" "${CONFIG_STEM}" "$@"
