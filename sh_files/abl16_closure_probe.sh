#!/usr/bin/env bash
#SBATCH --job-name=abl16-closure
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# exp16: предпроверка «замыкание или свободный источник» на ОБУЧЕННОМ чекпоинте R3.
#
# Ничего не обучает. Смотрит, с чем скоррелирован выход Q_θ: с внутренними полями ядра
# (конденсационный сток, ω) — тогда это член замыкания и арм X1 должен выйти слабым;
# или со статической географией — тогда это свободный источник и X1 ≈ −3 %.
# Предсказания зарегистрированы заранее: docs/architecture.md §7.6.
#
# GPU не нужен: сетка 32×64. Партиция cpu-e-quick не принимает --mem — не задавать.
#
# Запуск:  cd ~/wt_fix_v2 && sbatch sh_files/abl16_closure_probe.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME=pi-iamvp

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_fix_v2}"
export REPO_ROOT
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

CKPT="$(ls -t "${HOME}"/abl16_long_ckpt/abl16L-r3-a2-exp13-t12-s0/*/last.pt | head -1)"
MEMMAP="${HOME}/era5_memmap/predformer_usa_2000_2004.dat"
OUT_DIR="${HOME}/abl16_closure"
SCRIPT="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/closure/closure_probe.py"
mkdir -p "${OUT_DIR}"

echo "[closure] чекпоинт: ${CKPT}"
python "${SCRIPT}" \
  --checkpoint "${CKPT}" \
  --memmap "${MEMMAP}" \
  --out "${OUT_DIR}/closure_r3.npz" \
  --max-samples "${CLOSURE_MAX_SAMPLES:-256}"
echo "[closure] готово"
