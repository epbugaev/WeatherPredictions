#!/bin/bash
#SBATCH --job-name=exp22-metrics-prep
#SBATCH --account=proj_1715
#SBATCH --partition=cpu-e-quick
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Климатология (база отсчёта ACC, исключает val-2004) и пороги CSI/FSS (квантили
# ИСТИНЫ на 2004) на каждый регион exp22 — из packed-мемапа 2000-2004. Их читает
# харнесс метрик (ядро exp16). Скрипты — общие с exp16/20/21, здесь только пути.
# Чистый numpy, поэтому окружение predformer (в pi-iamvp тоже сработает).
set -euo pipefail
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-predformer}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/wt_exp21}}"
export REPO_ROOT
# shellcheck source=sh_files/_shell_contract.sh
source "${REPO_ROOT}/sh_files/_shell_contract.sh" "${REPO_ROOT}/sh_files"

MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
EXP16_METRICS="${REPO_ROOT}/docs/experiments/16_model_ablation_ladder/metrics"
read -r -a REGIONS <<<"${EXP22_REGIONS:-france npac}"

for region in "${REGIONS[@]}"; do
  memmap="${MEMMAP_DIR}/predformer_${region}_2000_2004.dat"
  if [[ ! -f "${memmap}" ]]; then
    echo "[exp22-prep] нет мемапа ${memmap} — сперва репак" >&2
    exit 2
  fi
  echo "[exp22-prep] ${region}: климатология (без val-2004)"
  python "${EXP16_METRICS}/climatology.py" \
    --memmap "${memmap}" \
    --out "${MEMMAP_DIR}/climatology_${region}_2000_2003.npz" \
    --val-year 2004
  echo "[exp22-prep] ${region}: пороги CSI/FSS на 2004"
  python "${EXP16_METRICS}/thresholds.py" \
    --memmap "${memmap}" \
    --out "${MEMMAP_DIR}/thresholds_${region}_2004.npz" \
    --val-year 2004
done
echo "[exp22-prep] готово -> climatology_{${REGIONS[*]}}_2000_2003.npz + thresholds_*_2004.npz"
