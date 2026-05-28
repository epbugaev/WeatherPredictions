#!/bin/bash
#SBATCH --job-name=smoke-simvp-usa-2gpu
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
export BENCH_MAX_STEPS="${BENCH_MAX_STEPS:-5}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"

exec bash "${weatherpred__launcher}" simvp_usa "$@"
