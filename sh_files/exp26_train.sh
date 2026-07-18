#!/bin/bash
# Волна обучающих джоб exp26 — статические входы exp24 на France и North Pacific.
# Регион берётся из имени конфига (…_france.yaml / …_npac.yaml); соответствующий
# packed-мемап выбирается автоматически. SKIP_STAGE=1 читает мемап на месте с
# shared-FS (безопасно для волны — стейджинг 25 ГБ в /tmp переполнял ноду).
#
# Модель берётся из model.type конфига (лончер модель-агностичный), поэтому один
# скрипт запускает любое из трёх семейств. Тяжёлым (IAM4VP/PredRNNv2) — NGPU=2.
#
#   ACCOUNT=proj_1717 NGPU=2 EXP26_JOBS="exp26_iam4vp_nophys_orog_france \
#     exp26_iam4vp_a2_orog_france" bash sh_files/exp26_train.sh
set -euo pipefail

# Дефолтный env контракта (weatherpred-gft-fix) сломан (GLIBC_2.28); pi-iamvp несёт
# весь train-стек. Экспортим, чтобы sbatch пробросил его в джобу (--export=ALL).
export WEATHERPRED_CONDA_ENV_NAME="${WEATHERPRED_CONDA_ENV_NAME:-pi-iamvp}"

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_exp26}"
MEMMAP_DIR="${MEMMAP_DIR:-${HOME}/era5_memmap}"
ACCOUNT="${ACCOUNT:-proj_1717}"
NGPU="${NGPU:-1}"
read -r -a JOBS <<<"${EXP26_JOBS:?Set EXP26_JOBS to config stems (space-separated)}"

for stem in "${JOBS[@]}"; do
  cfg="${REPO_ROOT}/configs/exp26/${stem}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[exp26-train] нет конфига ${cfg}" >&2
    exit 1
  fi
  case "${stem}" in
    *_france) region=france ;;
    *_npac) region=npac ;;
    *) echo "[exp26-train] в имени ${stem} нет региона (_france/_npac)" >&2; exit 1 ;;
  esac
  memmap="${MEMMAP_DIR}/predformer_${region}_2000_2004.dat"
  if [[ ! -f "${memmap}" ]]; then
    echo "[exp26-train] нет мемапа ${memmap} — сперва репак" >&2
    exit 1
  fi
  echo "[exp26-train] submit ${stem} region=${region} ngpu=${NGPU}"
  SKIP_STAGE=1 ORIG_MEMMAP="${memmap}" REPO_ROOT="${REPO_ROOT}" \
    sbatch -A "${ACCOUNT}" -J "${stem}" --gres="gpu:${NGPU}" \
    "${REPO_ROOT}/sh_files/train_v4_memmap.sh" "configs/exp26/${stem}.yaml"
done
