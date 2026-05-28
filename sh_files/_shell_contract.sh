#!/usr/bin/env bash
# Внутренний контракт окружения для launcher'ов и Slurm job-скриптов WeatherPredictions.
#
# Использование (из вызывающего скрипта, уже после set -euo pipefail):
#   _sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
#   # shellcheck source=sh_files/_shell_contract.sh
#   source "${_sc_here}/_shell_contract.sh" "${_sc_here}"
#
# Переменные окружения (все опциональны):
#   REPO_ROOT      — корень клона; иначе SLURM_SUBMIT_DIR либо авто от расположения скрипта.
#   CONDA_ENV_BIN  — абсолютный путь к .../envs/weatherpred-gft-fix/bin; если задан, prepend в PATH.
#                    Иначе на Slurm: ``module load Python/Miniconda_v25`` + ``conda activate weatherpred-gft-fix``.
#   .env           — если ${REPO_ROOT}/.env существует, экспортируется перед активацией окружения.
#
# Якорь репозитория — наличие train.py в корне (канонический entrypoint).

weatherpred__shell_contract_anchor="${1:?_shell_contract: передайте каталог скрипта-носителя}"

REPO_ROOT="${REPO_ROOT:-}"
if [[ -n "${REPO_ROOT}" ]]; then
  if [[ ! -f "${REPO_ROOT}/train.py" ]]; then
    echo "[_shell_contract] REPO_ROOT задан, но нет train.py: ${REPO_ROOT}" >&2
    return 2 2>/dev/null || exit 2
  fi
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  # Скрипт-носитель лежит в sh_files/ → REPO_ROOT одной директорией выше.
  REPO_ROOT="$(cd "${weatherpred__shell_contract_anchor}/.." && pwd)"
fi

if [[ ! -f "${REPO_ROOT}/train.py" ]]; then
  echo "[_shell_contract] Не найден train.py (REPO_ROOT=${REPO_ROOT:-<пусто>}). Задайте REPO_ROOT или sbatch из корня репозитория." >&2
  return 2 2>/dev/null || exit 2
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/.env"
  set +a
  echo "[_shell_contract] Загружен .env из ${REPO_ROOT}/.env" >&2
fi

export REPO_ROOT
cd "${REPO_ROOT}" || return 1 2>/dev/null || exit 1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${REPO_ROOT}/logs"

if [[ -z "${CONDA_ENV_BIN:-}" && -x /home/ebugaev/.conda/envs/weatherpred-gft-fix/bin/python ]]; then
  CONDA_ENV_BIN="/home/ebugaev/.conda/envs/weatherpred-gft-fix/bin"
fi

if [[ -n "${CONDA_ENV_BIN:-}" ]]; then
  export PATH="${CONDA_ENV_BIN%/}:${PATH}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh
  fi
  if ! type module >/dev/null 2>&1; then
    echo "[_shell_contract] Ошибка: CONDA_ENV_BIN не задан и команда module недоступна; задайте CONDA_ENV_BIN явно." >&2
    return 2 2>/dev/null || exit 2
  fi
  module purge
  module load Python/Miniconda_v25
  if ! command -v conda >/dev/null 2>&1; then
    echo "[_shell_contract] Ошибка: после module load Python/Miniconda_v25 команда conda недоступна." >&2
    return 2 2>/dev/null || exit 2
  fi
  weatherpred__activate_conda_fallback() {
    # ``source _shell_contract.sh <anchor>`` подменяет positional args текущего shell.
    # Запускаем activate внутри функции без аргументов, чтобы conda не восприняла
    # ``<anchor>`` как имя окружения и при этом не испортила ``$1`` launcher'а.
    # shellcheck disable=SC1091
    source "$(conda info --base)/bin/activate"
    conda activate weatherpred-gft-fix
  }
  weatherpred__activate_conda_fallback
  unset -f weatherpred__activate_conda_fallback
  echo "[_shell_contract] CONDA_ENV_BIN не задан; активировано conda-окружение weatherpred-gft-fix через Python/Miniconda_v25." >&2
fi

unset weatherpred__shell_contract_anchor
