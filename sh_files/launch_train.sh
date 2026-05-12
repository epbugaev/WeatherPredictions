#!/usr/bin/env bash
# =============================================================================
# Канонический public entrypoint обучения WeatherPredictions.
#
# Назначение
# ----------
# Резолвит окружение (``_shell_contract.sh``) и один ``exec`` на ``train.py``.
# Первый позиционный аргумент — стем YAML-конфига (``configs/<stem>.yaml``)
# либо явный путь (содержит ``/`` или оканчивается на ``.yaml``).
# Остальные позиционные аргументы пробрасываются в ``train.py`` как есть.
#
# Использование
# -------------
#   sbatch sh_files/<start_*.sh>                          # из job-скрипта
#   bash   sh_files/launch_train.sh predformer_usa        # локально / интерактивно
#   bash   sh_files/launch_train.sh configs/simvp.yaml    # явный путь
#
# Параметры распределённого запуска
# ---------------------------------
# NNODES / NGPUS читаются из env, иначе из SLURM_JOB_NUM_NODES / SLURM_GPUS_ON_NODE,
# иначе по 1. Переопределение — через env-переменные перед sbatch.
#
# Запуск происходит через ``torchrun``: он раскладывает RANK/LOCAL_RANK/WORLD_SIZE
# в окружение, ``utils.distributed.setup_distributed`` их читает.
# MASTER_ADDR/MASTER_PORT для multi-node берутся из env, для одного узла подставляем
# 127.0.0.1:29500.
#
# Окружение: sh_files/_shell_contract.sh
# =============================================================================
set -euo pipefail

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

CONFIG_ARG="${1:-}"
if [[ -z "${CONFIG_ARG}" ]]; then
  echo "Usage: $0 <config_stem|config_path> [extra args for train.py]" >&2
  echo "Пример: $0 predformer_usa" >&2
  exit 1
fi
shift

if [[ "${CONFIG_ARG}" == *.yaml || "${CONFIG_ARG}" == */* ]]; then
  CONFIG_PATH="${CONFIG_ARG}"
else
  CONFIG_PATH="configs/${CONFIG_ARG}.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[launch_train] Конфиг не найден: ${CONFIG_PATH}" >&2
  exit 1
fi

NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}"

echo "[launch_train] JobID=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd)"
echo "[launch_train] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[launch_train] config=${CONFIG_PATH} nodes=${NNODES} gpus_per_node=${NGPUS}"
echo "[launch_train] python=$(command -v python)"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec python -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NGPUS}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  train.py --config "${CONFIG_PATH}" "$@"
