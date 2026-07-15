#!/bin/bash
# Волна обучающих джоб exp22. Каждый конфиг несёт регион в имени файла
# (…_france.yaml / …_npac.yaml); соответствующий packed-мемап выбирается
# автоматически. SKIP_STAGE=1 читает мемап на месте с shared-FS (безопасно для
# волны — стейджинг 25 ГБ в /tmp на каждую джобу переполнял ноду в exp20).
#
# Модель берётся из model.type конфига (лончер модель-агностичный), поэтому один
# скрипт запускает любое из трёх семейств. Тяжёлым (IAM4VP/PredRNNv2) можно дать
# 2 GPU через NGPU=2.
#
#   EXP22_JOBS="exp22_simvpv2_nophys_france exp22_simvpv2_a2_france \
#     exp22_simvpv2_legacy_france exp22_simvpv2_nophys_npac \
#     exp22_simvpv2_a2_npac exp22_simvpv2_legacy_npac" bash sh_files/exp22_train.sh
set -euo pipefail

# Дефолтный env контракта (weatherpred-gft-fix) сломан (GLIBC_2.28); pi-iamvp несёт
# весь train-стек (torch 2.6 + comet + Data/Models/strategies). Экспортим, чтобы
# sbatch пробросил его в джобу (--export=ALL) → _shell_contract активирует его.
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_exp21}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
ACCOUNT="${ACCOUNT:-proj_1715}"
NGPU="${NGPU:-1}"
read -r -a JOBS <<<"${EXP22_JOBS:?Set EXP22_JOBS to config stems (space-separated)}"

for stem in "${JOBS[@]}"; do
  cfg="${REPO_ROOT}/configs/exp22/${stem}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[exp22-train] нет конфига ${cfg}" >&2
    exit 1
  fi
  case "${stem}" in
    *_france) region=france ;;
    *_npac) region=npac ;;
    *) echo "[exp22-train] в имени ${stem} нет региона (_france/_npac)" >&2; exit 1 ;;
  esac
  memmap="${MEMMAP_DIR}/predformer_${region}_2000_2004.dat"
  if [[ ! -f "${memmap}" ]]; then
    echo "[exp22-train] нет мемапа ${memmap} — сперва репак" >&2
    exit 1
  fi
  echo "[exp22-train] submit ${stem} region=${region} ngpu=${NGPU}"
  SKIP_STAGE=1 ORIG_MEMMAP="${memmap}" REPO_ROOT="${REPO_ROOT}" \
    sbatch -A "${ACCOUNT}" -J "${stem}" --gres="gpu:${NGPU}" \
    "${REPO_ROOT}/sh_files/train_v4_memmap.sh" "configs/exp22/${stem}.yaml"
done
