#!/usr/bin/env bash
# =============================================================================
# Локальный helper: push текущей ветки → pull на cHARISMa → sbatch.
#
# Использование
# -------------
#   bash sh_files/remote_submit.sh sh_files/train_PredFormer_USA.sh
#   bash sh_files/remote_submit.sh -f sh_files/train_SimVp_USA.sh   # +tail логов
#   bash sh_files/remote_submit.sh --allow-dirty sh_files/train_PredFormer_USA.sh
#
# Переменные окружения
# --------------------
#   REMOTE_HOST   — SSH-алиас (default: cluster)
#   REMOTE_REPO   — путь к клону на кластере (default: /home/fa.buzaev/WeatherPredictions)
#
# Flow
# ----
#   1) Проверка: нет uncommitted changes (или --allow-dirty).
#   2) git push origin HEAD:<branch>.
#   3) ssh ${REMOTE_HOST}: cd ${REMOTE_REPO} && fetch + checkout + pull --ff-only && sbatch.
#   4) Парсим JobID; при -f tail-им logs/slurm-<jobname>-<jobid>.{out,err}.
# =============================================================================
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cluster}"
REMOTE_REPO="${REMOTE_REPO:-/home/fa.buzaev/WeatherPredictions}"

FOLLOW=0
ALLOW_DIRTY=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--follow)    FOLLOW=1; shift ;;
    --allow-dirty)  ALLOW_DIRTY=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    --) shift; POSITIONAL+=("$@"); break ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "Usage: $0 [-f|--follow] [--allow-dirty] <sbatch_script>" >&2
  exit 1
fi
SCRIPT="${POSITIONAL[0]}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[remote_submit] sbatch-скрипт не найден локально: ${SCRIPT}" >&2
  exit 1
fi

BRANCH="$(git symbolic-ref --short HEAD)"
if [[ -z "${BRANCH}" ]]; then
  echo "[remote_submit] HEAD в detached-состоянии; переключись на именованную ветку." >&2
  exit 1
fi

if [[ ${ALLOW_DIRTY} -eq 0 ]]; then
  if ! git diff --quiet HEAD || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "[remote_submit] Есть незакоммиченные изменения. Закоммить/стэшни, либо передай --allow-dirty (тогда они НЕ попадут на кластер)." >&2
    git status --short >&2
    exit 1
  fi
fi

echo "[remote_submit] push ${BRANCH} → origin"
git push origin "HEAD:refs/heads/${BRANCH}"

echo "[remote_submit] sync & sbatch on ${REMOTE_HOST}:${REMOTE_REPO}"
SUBMIT_OUT="$(ssh "${REMOTE_HOST}" bash -se <<EOF
set -euo pipefail
cd "${REMOTE_REPO}"
git fetch --prune origin
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "${BRANCH}"
else
  git checkout -b "${BRANCH}" "origin/${BRANCH}"
fi
git pull --ff-only origin "${BRANCH}"
sbatch "${SCRIPT}"
EOF
)"
echo "${SUBMIT_OUT}"

JOB_ID="$(echo "${SUBMIT_OUT}" | awk '/Submitted batch job/ {print $NF}')"
if [[ -z "${JOB_ID}" ]]; then
  echo "[remote_submit] Не удалось распарсить JobID из вывода sbatch." >&2
  exit 1
fi

JOB_NAME="$(grep -E '^#SBATCH[[:space:]]+--job-name' "${SCRIPT}" | head -1 | sed -E 's/.*--job-name[[:space:]]*=?[[:space:]]*//' | tr -d '"')"
JOB_NAME="${JOB_NAME:-$(basename "${SCRIPT}" .sh)}"

echo "[remote_submit] JobID=${JOB_ID} job-name=${JOB_NAME}"
echo "[remote_submit] logs: ${REMOTE_REPO}/logs/slurm-${JOB_NAME}-${JOB_ID}.{out,err}"
echo "[remote_submit] status: ssh ${REMOTE_HOST} squeue -j ${JOB_ID}"

if [[ ${FOLLOW} -eq 1 ]]; then
  echo "[remote_submit] tailing logs (Ctrl-C для detach; job продолжит работать):"
  exec ssh "${REMOTE_HOST}" "cd ${REMOTE_REPO} && \
    until [[ -f logs/slurm-${JOB_NAME}-${JOB_ID}.out ]]; do sleep 2; done && \
    tail -F logs/slurm-${JOB_NAME}-${JOB_ID}.out logs/slurm-${JOB_NAME}-${JOB_ID}.err"
fi
