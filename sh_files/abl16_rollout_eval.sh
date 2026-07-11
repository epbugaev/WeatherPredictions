#!/bin/bash
#SBATCH --job-name=abl16-rollout
#SBATCH --account=proj_1715
#SBATCH --partition=rocky,gpu-ef-quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# 12-шаговый rollout-инференс (free-running + teacher-forced) всех армов exp16
# с чекпоинтов общей эпохи (tag epoch=17 == эпоха 18). Пишет npz на арм в
# ${OUT_DIR}. Армы R2a/R2 обучены из pre-exp13 worktree: их инференс идёт
# кодом того дерева (REPO_ROOT=${PREFIX13_ROOT} на вызов), сам скрипт один.
set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_fix_v2}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT_BASE="${CKPT_BASE:-${HOME}/abl16_ckpt}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2000_2004.dat}"
OUT_DIR="${OUT_DIR:-${HOME}/abl16_rollout}"
EPOCH_TAG="${EPOCH_TAG:-17}"
PREFIX13_ROOT="${PREFIX13_ROOT:-${HOME}/wt_prefix13}"
SCRIPT="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/rollout_eval.py"
mkdir -p "${OUT_DIR}"

ARMS=(
  abl16-r0-no-physics-s0
  abl16-r1-legacy-hybrid-s0
  abl16-r2a-a1-pre13-s0
  abl16-r2-a2-pre13-s0
  abl16-r3-a2-exp13-s0
  abl16-r4-exp14-s0
  abl16-r5-exp15-s0
)

for arm in "${ARMS[@]}"; do
  ckpt="$(ls -t "${CKPT_BASE}/${arm}"/*/epoch="${EPOCH_TAG}"-*.pt | head -1)"
  arm_repo="${REPO_ROOT}"
  if [[ "${arm}" == *pre13* ]]; then
    arm_repo="${PREFIX13_ROOT}"
  fi
  echo "[abl16-rollout] arm=${arm} repo=${arm_repo} ckpt=${ckpt}"
  REPO_ROOT="${arm_repo}" python "${SCRIPT}" \
    --checkpoint "${ckpt}" \
    --memmap "${MEMMAP}" \
    --out "${OUT_DIR}/rollout_${arm}.npz"
done
echo "[abl16-rollout] all arms done -> ${OUT_DIR}"
