#!/bin/bash
#SBATCH --job-name=train-pi-predrnnv2
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# v4 raw-memmap training for the exp20 PI-PredRNNv2 ablation ladder.
# Pass the config as the first positional argument, e.g.
#   sh_files/train_pi_predrnnv2.sh configs/exp20/exp20_p3_a2_exp13_s0.yaml
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

weatherpred__repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
weatherpred__env_file="${weatherpred__repo_root}/.env"
if [[ -f "${weatherpred__env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${weatherpred__env_file}"
  set +a
  weatherpred__repo_root="${REPO_ROOT:-${weatherpred__repo_root}}"
fi
weatherpred__launcher="${weatherpred__repo_root}/sh_files/launch_train.sh"
if [[ ! -f "${weatherpred__launcher}" ]]; then
  weatherpred__launcher="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/launch_train.sh"
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"

# Staging copies a 24 GB memmap to node-local NVMe. That is the fast path, but
# several concurrent jobs on one node need 24 GB of /tmp EACH, and a node
# already hosting other waves runs out ("No space left on device", jobs 4175559
# /4175560). SKIP_STAGE=1 reads the memmap straight off the shared filesystem —
# slower per batch, but immune to a crowded node.
if [[ "${SKIP_STAGE:-0}" == "1" ]]; then
  echo "[train-v4] SKIP_STAGE=1: reading memmap in place from ${ORIG_MEMMAP}"
  export MEMMAP_PATH_OVERRIDE="${ORIG_MEMMAP}"
  export MEMMAP_META_PATH_OVERRIDE="${weatherpred__orig_meta}"
else
  STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
  mkdir -p "${STAGE_DIR}"
  echo "[train-v4] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
  weatherpred__t0=$(date +%s)
  cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
  if [[ -f "${weatherpred__orig_meta}" ]]; then
    cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
  fi
  echo "[train-v4] staged in $(( $(date +%s) - weatherpred__t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
  export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
  export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${weatherpred__orig_meta}")"
  trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT
fi

weatherpred__config="${1:-configs/exp20/exp20_p0_no_physics_s0.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
exec bash "${weatherpred__launcher}" "${weatherpred__config}" "$@"
