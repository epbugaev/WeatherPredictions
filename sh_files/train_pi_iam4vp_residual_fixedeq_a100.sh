#!/bin/bash
# =============================================================================
# A100 retraining of PI-IAM4VP-ResidualStablePhysics on the FIXED inline physics
# (branch fix_inline_equations). Produces experiment
# "PI-IAM4VP-ResidualStablePhysics-USA-v4-FIXEDEQ" for A/B comparison against the
# original buggy-physics run "PI-IAM4VP-ResidualStablePhysics-USA-v4".
#
# Submit from the FIXED copy root:
#   cd /home/fa.buzaev/WeatherPredictions
#   sbatch sh_files/train_pi_iam4vp_residual_fixedeq_a100.sh
# =============================================================================
#SBATCH --job-name=pi-iam4vp-resid-fixedeq
#SBATCH --partition=rocky
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="type_e|type_f|type_h"
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

# --- pin to the FIXED copy explicitly (do not trust cwd) ---
export REPO_ROOT="/home/fa.buzaev/WeatherPredictions"
cd "${REPO_ROOT}"

# --- verify we are actually on the fix branch with the patched physics ---
_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "[fixedeq] repo=${REPO_ROOT} branch=${_branch} sha=${_sha} host=$(hostname)"
if ! grep -q "f_field = (2 \* 7.2921e-5" Models/WeatherGFT.py; then
  echo "[fixedeq] ERROR: patched Coriolis f_field not found in Models/WeatherGFT.py — wrong checkout?" >&2
  exit 3
fi

# --- checkpoints to a writable location; keep runs separated by exp name ---
export CHECKPOINT_BASE_OVERRIDE="/home/fa.buzaev/checkpoints"
mkdir -p "${CHECKPOINT_BASE_OVERRIDE}" logs

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_PORT="${MASTER_PORT:-$((29000 + ${SLURM_JOB_ID:-0} % 1000))}"

# --- stage the 24GB USA memmap to node-local /tmp (shared /home is slow for random reads) ---
ORIG_MEMMAP="${WEATHERPRED_USA_MEMMAP:-/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat}"
STAGE_DIR="/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}"
mkdir -p "${STAGE_DIR}"
echo "[fixedeq] staging ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
t0=$(date +%s)
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
[[ -f "${ORIG_MEMMAP%.dat}.meta.json" ]] && cp "${ORIG_MEMMAP%.dat}.meta.json" "${STAGE_DIR}/"
echo "[fixedeq] staged in $(( $(date +%s) - t0 ))s: $(du -sh "${STAGE_DIR}" | cut -f1)"
export MEMMAP_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"
export MEMMAP_META_PATH_OVERRIDE="${STAGE_DIR}/$(basename "${ORIG_MEMMAP%.dat}.meta.json")"
trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

# --- launch (single A100) via the canonical launcher; NGPUS=1 ---
export NGPUS=1 NNODES=1
exec bash "${REPO_ROOT}/sh_files/launch_train.sh" pi_iam4vp_residual_usa_v4_fixedeq
