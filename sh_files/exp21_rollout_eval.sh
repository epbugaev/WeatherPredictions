#!/bin/bash
#SBATCH --job-name=exp21-rollout
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# 12-шаговый rollout-инференс всех армов лестницы exp21 (PI-SimVPv2) с общего
# чекпоинта last.pt (эпоха 500). Пишет по npz на арм в ${OUT_DIR}. SimVPv2 — MIMO:
# весь горизонт одним форвардом, teacher-forcing не определён (rmse_forced=rmse_free),
# поэтому режим один. Считаем на CPU: cpu-e-quick несравнимо менее загружен, чем GPU.
# Локально по npz рисует docs/experiments/21_pi_simvpv2_ladder/rollout_figures.py.
set -euo pipefail
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp21}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT_BASE="${CKPT_BASE:-${HOME}/exp21_ckpt}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2004.dat}"
OUT_DIR="${OUT_DIR:-${HOME}/exp21_rollout}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
SCRIPT="${REPO_ROOT}/docs/experiments/21_pi_simvpv2_ladder/rollout_eval.py"
mkdir -p "${OUT_DIR}"

# Порядок = порядок лестницы (README §4): контроль S0, легаси, A-семейство (batched),
# контроль связки S0c, связка с физикой S3c. EXP21_ARMS позволяет досчитать подмножество
# (армы одной эпохи 500 — ранжирование на разных эпохах невалидно, exp16 §11.3).
read -r -a ARMS <<<"${EXP21_ARMS:-s0_no_physics s1_legacy_hybrid s3a_no_diabatic s3_a2_exp13 s4_exp14 s5_exp15 s0c_no_physics_chained s3c_a2_exp13_chained}"

for arm in "${ARMS[@]}"; do
  run_name="exp21L-${arm//_/-}-t12-seed0"
  ckpt="$(ls -t "${CKPT_BASE}/${run_name}"/*/"${CKPT_NAME}" | head -1)"
  echo "[exp21-rollout] arm=${run_name} ckpt=${ckpt}"
  python "${SCRIPT}" \
    --checkpoint "${ckpt}" \
    --memmap "${MEMMAP}" \
    --out "${OUT_DIR}/rollout_${run_name}.npz" \
    --num-workers 8
done
echo "[exp21-rollout] all arms done -> ${OUT_DIR}"
