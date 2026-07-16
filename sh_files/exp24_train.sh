#!/bin/bash
# Волна обучающих джоб exp24 (USA, статические входы). Модель берётся из
# model.type конфига; один скрипт запускает любое из трёх семейств.
# Тяжёлым (IAM4VP/PredRNNv2) можно дать 2 GPU через NGPU=2.
#
#   EXP24_JOBS="exp24_simvpv2_nophys_static_usa exp24_simvpv2_a2_static_usa" \
#     bash sh_files/exp24_train.sh
set -euo pipefail

# Дефолтный env контракта (weatherpred-gft-fix) сломан (GLIBC_2.28); pi-iamvp несёт
# весь train-стек. Экспортим, чтобы sbatch пробросил его в джобу.
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_exp24}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2000_2004.dat}"
ACCOUNT="${ACCOUNT:-proj_1715}"
NGPU="${NGPU:-1}"
read -r -a JOBS <<<"${EXP24_JOBS:?Set EXP24_JOBS to config stems (space-separated)}"

if [[ ! -f "${MEMMAP}" ]]; then
  echo "[exp24-train] нет мемапа ${MEMMAP}" >&2
  exit 1
fi

for stem in "${JOBS[@]}"; do
  cfg="${REPO_ROOT}/configs/exp24/${stem}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[exp24-train] нет конфига ${cfg}" >&2
    exit 1
  fi
  echo "[exp24-train] submit ${stem} ngpu=${NGPU}"
  SKIP_STAGE=1 ORIG_MEMMAP="${MEMMAP}" REPO_ROOT="${REPO_ROOT}" \
    sbatch -A "${ACCOUNT}" -J "${stem}" --gres="gpu:${NGPU}" \
    "${REPO_ROOT}/sh_files/train_v4_memmap.sh" "configs/exp24/${stem}.yaml"
done
