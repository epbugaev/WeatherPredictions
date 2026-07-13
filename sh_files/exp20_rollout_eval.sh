#!/bin/bash
#SBATCH --job-name=exp20-rollout
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# 12-шаговый rollout-инференс (free-running + teacher-forced) всех армов
# лестницы exp20 (PI-PredRNNv2) с общего чекпоинта last.pt (эпоха 40). Пишет
# по npz на арм в ${OUT_DIR}. Считаем на CPU: горизонт нативный (один форвард
# на сэмпл), а cpu-e-quick несравнимо менее загружен, чем GPU-очереди.
set -euo pipefail
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp20}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT_BASE="${CKPT_BASE:-${HOME}/exp20_ckpt}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2004.dat}"
OUT_DIR="${OUT_DIR:-${HOME}/exp20_rollout}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
SCRIPT="${REPO_ROOT}/docs/experiments/20_pi_predrnnv2_ladder/rollout_eval.py"
mkdir -p "${OUT_DIR}"

# Порядок = порядок лестницы (README §4): контроль, легаси, затем A-семейство.
ARMS=(
  p0_no_physics
  p1_legacy_hybrid
  p3a_no_diabatic
  p3_a2_exp13
  p4_exp14
  p5_exp15
)

for arm in "${ARMS[@]}"; do
  run_name="exp20-${arm//_/-}-s0"
  ckpt="$(ls -t "${CKPT_BASE}/${run_name}"/*/"${CKPT_NAME}" | head -1)"
  echo "[exp20-rollout] arm=${run_name} ckpt=${ckpt}"
  python "${SCRIPT}" \
    --checkpoint "${ckpt}" \
    --memmap "${MEMMAP}" \
    --out "${OUT_DIR}/rollout_${run_name}.npz" \
    --num-workers 8
done
echo "[exp20-rollout] all arms done -> ${OUT_DIR}"
