#!/bin/bash
# Волна обучающих джоб exp25 (изоляция маршрута орографии, USA, IAM4VP A2).
# Модель берётся из model.type конфига. IAM4VP тяжёлый — обычно NGPU=2.
# d1 НЕ обучается (переиспускается abl16-r3 @500); запускаем d0/d2/d3.
#
#   EXP25_JOBS="exp25_iam4vp_d0_usa exp25_iam4vp_d2_usa exp25_iam4vp_d3_usa" \
#     NGPU=2 bash sh_files/exp25_train.sh
set -euo pipefail

# Дефолтный env контракта (weatherpred-gft-fix) сломан (GLIBC_2.28); pi-iamvp несёт
# весь train-стек. Экспортим, чтобы sbatch пробросил его в джобу.
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_exp25}"
MEMMAP="${MEMMAP:-${HOME}/era5_memmap/predformer_usa_2000_2004.dat}"
ACCOUNT="${ACCOUNT:-proj_1717}"
NGPU="${NGPU:-2}"
read -r -a JOBS <<<"${EXP25_JOBS:?Set EXP25_JOBS to config stems (space-separated)}"

if [[ ! -f "${MEMMAP}" ]]; then
  echo "[exp25-train] нет мемапа ${MEMMAP}" >&2
  exit 1
fi

for stem in "${JOBS[@]}"; do
  cfg="${REPO_ROOT}/configs/exp25/${stem}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[exp25-train] нет конфига ${cfg}" >&2
    exit 1
  fi
  echo "[exp25-train] submit ${stem} ngpu=${NGPU}"
  SKIP_STAGE=1 ORIG_MEMMAP="${MEMMAP}" REPO_ROOT="${REPO_ROOT}" \
    sbatch -A "${ACCOUNT}" -J "${stem}" --gres="gpu:${NGPU}" \
    "${REPO_ROOT}/sh_files/train_v4_memmap.sh" "configs/exp25/${stem}.yaml"
done
