#!/usr/bin/env bash
# =============================================================================
# Dataloader bench submit script (Шаг 0 / Шаг 1 из dataloader-плана).
#
# Назначение
# ----------
# Запускает короткий tренировочный «эпизод» с включённым ``torch.profiler``
# для диагностики GPU-idle. Берёт ``configs/bench_predformer_usa.yaml`` (по
# умолчанию) и выставляет два env-флага, которые читает ``trainer.py``:
#   BENCH_MAX_STEPS  — жёсткий верхний предел шагов обучения (default 60);
#   PROFILE_TRACE_DIR — путь, куда дампится chrome-tracing JSON и dmon.csv.
# Тренер при этих флагах не запускает validation / checkpoint и выходит после
# первого epoch — поэтому продакшен-поведение остаётся неизменным, когда
# флаги не выставлены.
#
# Использование
# -------------
#   sbatch sh_files/bench_dataloader.sh                          # default cold-run
#   BENCH_TAG=step1_warm sbatch sh_files/bench_dataloader.sh     # повтор для warm-cache
#   BENCH_CONFIG=bench_simvp_usa sbatch sh_files/bench_dataloader.sh
#
# Артефакты пишутся в ``bench_logs/<TS>_<host>_<jobid>_<tag>/``:
#   trace.json (torch.profiler chrome-tracing), dmon.csv (nvidia-smi),
#   bench.log (stdout/stderr), config_snapshot.yaml, git_sha.txt.
# =============================================================================
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --constraint="type_e|type_f"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=dl_bench
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Resolve repo + activate environment via the project-wide contract.
# SLURM copies the batch script into /var/spool/slurmd/jobNNN/, so
# $BASH_SOURCE points at the spool copy — relying on $(dirname $0) breaks.
# Use $SLURM_SUBMIT_DIR (where ``sbatch`` was issued; SLURM also makes it the
# job's initial CWD) when present, fall back to $BASH_SOURCE for interactive
# ``bash sh_files/bench_dataloader.sh`` runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _sc_here="${SLURM_SUBMIT_DIR}/sh_files"
else
  _sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

BENCH_CONFIG="${BENCH_CONFIG:-bench_predformer_usa}"
BENCH_TAG="${BENCH_TAG:-step1_cold}"
export BENCH_MAX_STEPS="${BENCH_MAX_STEPS:-60}"

CFG_PATH="configs/${BENCH_CONFIG}.yaml"
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[bench] config not found: ${CFG_PATH} (cwd=$(pwd) REPO_ROOT=${REPO_ROOT:-<unset>})" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
HOSTNAME_SHORT="$(hostname -s)"
JOBID="${SLURM_JOB_ID:-local}"
TRACE_DIR="${REPO_ROOT}/bench_logs/${TS}_${HOSTNAME_SHORT}_${JOBID}_${BENCH_TAG}"
mkdir -p "${TRACE_DIR}"
export PROFILE_TRACE_DIR="${TRACE_DIR}"

cp "${CFG_PATH}" "${TRACE_DIR}/config_snapshot.yaml"
git rev-parse HEAD > "${TRACE_DIR}/git_sha.txt" 2>/dev/null || echo unknown > "${TRACE_DIR}/git_sha.txt"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Avoid port clash with concurrent production launch_train (29500).
export MASTER_PORT="${MASTER_PORT:-29501}"

NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}"

# Background GPU sampler. ``dmon`` outputs one row per second per GPU.
nvidia-smi dmon -s pucvmet -d 1 -o T > "${TRACE_DIR}/dmon.csv" 2>&1 &
DMON_PID=$!
trap 'kill ${DMON_PID} 2>/dev/null || true' EXIT

echo "[bench] tag=${BENCH_TAG} config=${BENCH_CONFIG} steps=${BENCH_MAX_STEPS} trace=${TRACE_DIR}"
echo "[bench] env: BENCH_MAX_STEPS=${BENCH_MAX_STEPS} PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR}"
echo "[bench] cwd=$(pwd) REPO_ROOT=${REPO_ROOT} host=$(hostname)"
echo "[bench] python=$(command -v python) git_sha=$(cat "${TRACE_DIR}/git_sha.txt")"

# Call torchrun inline (no need to re-source _shell_contract via launch_train.sh).
python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NGPUS}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    train.py --config "${CFG_PATH}" 2>&1 | tee "${TRACE_DIR}/bench.log"

echo "[bench] done: ${TRACE_DIR}"
ls -la "${TRACE_DIR}"
