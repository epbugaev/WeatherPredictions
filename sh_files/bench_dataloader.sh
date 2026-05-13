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

# Optional staging step: copy the netCDF tree for the configured year range
# from Lustre (/home/fratnikov/...) to local NVMe under /tmp, then patch the
# config snapshot to point ``data.input_folder`` at the staged path. Activate
# by setting ``STAGE_DATA=1`` (default off). ``STAGE_YEARS`` overrides the
# default 2000-2004 window. Used to test the §2.4 hypothesis: that 24
# small-file reads per sample on Lustre dominate ``data_wait``.
ACTIVE_CFG="${CFG_PATH}"
STAGE_DIR=""
if [[ "${STAGE_DATA:-0}" = "1" ]]; then
  STAGE_DIR="/tmp/${USER:-$(id -un)}/era5_stage_${JOBID}"
  mkdir -p "${STAGE_DIR}"
  STAGE_YEARS="${STAGE_YEARS:-2000 2001 2002 2003 2004}"
  STAGE_VARS="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind \
total_cloud_cover total_precipitation toa_incident_solar_radiation \
geopotential temperature specific_humidity relative_humidity \
u_component_of_wind v_component_of_wind vorticity potential_vorticity"
  STAGE_SRC="${STAGE_SRC:-/home/fratnikov/weather_bench/1.40625deg}"

  STAGE_START=$(date +%s)
  echo "[bench-stage] years=${STAGE_YEARS} src=${STAGE_SRC} dst=${STAGE_DIR}"
  for var in ${STAGE_VARS}; do
    (
      mkdir -p "${STAGE_DIR}/${var}"
      for year in ${STAGE_YEARS}; do
        src_file="${STAGE_SRC}/${var}/${var}_${year}_1.40625deg.nc"
        if [[ -f "${src_file}" ]]; then
          cp "${src_file}" "${STAGE_DIR}/${var}/"
        fi
      done
    ) &
  done
  wait
  STAGE_END=$(date +%s)
  STAGE_SECS=$((STAGE_END - STAGE_START))
  STAGE_SIZE=$(du -sh "${STAGE_DIR}" 2>/dev/null | cut -f1)
  echo "[bench-stage] done in ${STAGE_SECS}s, size=${STAGE_SIZE}"

  python3 - <<PYEOF
import yaml
snap = "${TRACE_DIR}/config_snapshot.yaml"
with open(snap) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("data", {})["input_folder"] = "${STAGE_DIR}/"
with open(snap, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
  ACTIVE_CFG="${TRACE_DIR}/config_snapshot.yaml"
fi

# Optional memmap staging: copy the v3_memmap .dat + .meta.json from wherever
# the bench config points (typically Lustre /home/fa.buzaev/era5_memmap/) to
# local NVMe under /tmp, then patch ``data.memmap_path`` in the config
# snapshot. Activate by setting ``STAGE_MEMMAP=1`` (default off). Used to
# isolate Lustre random-access overhead in the §2.5 path from the rest of
# the memmap pipeline.
if [[ "${STAGE_MEMMAP:-0}" = "1" ]]; then
  ORIG_MEMMAP_PATH=$(python3 - <<PYEOF
import yaml
cfg = yaml.safe_load(open("${ACTIVE_CFG}"))
print(cfg.get("data", {}).get("memmap_path", ""))
PYEOF
)
  if [[ -z "${ORIG_MEMMAP_PATH}" ]]; then
    echo "[bench-stage-memmap] data.memmap_path not set in ${ACTIVE_CFG}; aborting" >&2
    exit 1
  fi
  STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${JOBID}}"
  mkdir -p "${STAGE_DIR}"
  STAGE_START=$(date +%s)
  echo "[bench-stage-memmap] src=${ORIG_MEMMAP_PATH} dst=${STAGE_DIR}/"
  cp "${ORIG_MEMMAP_PATH}" "${STAGE_DIR}/"
  ORIG_META="${ORIG_MEMMAP_PATH%.dat}.meta.json"
  if [[ -f "${ORIG_META}" ]]; then
    cp "${ORIG_META}" "${STAGE_DIR}/"
  fi
  STAGE_SECS=$(( $(date +%s) - STAGE_START ))
  STAGE_SIZE=$(du -sh "${STAGE_DIR}" 2>/dev/null | cut -f1)
  echo "[bench-stage-memmap] done in ${STAGE_SECS}s, size=${STAGE_SIZE}"
  STAGED_MEMMAP="${STAGE_DIR}/$(basename "${ORIG_MEMMAP_PATH}")"
  python3 - <<PYEOF
import yaml
snap = "${ACTIVE_CFG}"
with open(snap) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("data", {})["memmap_path"] = "${STAGED_MEMMAP}"
with open(snap, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
fi

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
# Combined cleanup: kill dmon and (when staged) remove the /tmp copy so /tmp
# stays usable for subsequent jobs on this node.
trap 'kill ${DMON_PID} 2>/dev/null || true; [[ -n "${STAGE_DIR}" ]] && rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

echo "[bench] tag=${BENCH_TAG} config=${BENCH_CONFIG} steps=${BENCH_MAX_STEPS} trace=${TRACE_DIR}"
echo "[bench] env: BENCH_MAX_STEPS=${BENCH_MAX_STEPS} PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR}"
echo "[bench] cwd=$(pwd) REPO_ROOT=${REPO_ROOT} host=$(hostname)"
echo "[bench] python=$(command -v python) git_sha=$(cat "${TRACE_DIR}/git_sha.txt")"
echo "[bench] active_config=${ACTIVE_CFG} stage_data=${STAGE_DATA:-0} stage_dir=${STAGE_DIR:-<none>}"

# Call torchrun inline (no need to re-source _shell_contract via launch_train.sh).
python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NGPUS}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    train.py --config "${ACTIVE_CFG}" 2>&1 | tee "${TRACE_DIR}/bench.log"

echo "[bench] done: ${TRACE_DIR}"
ls -la "${TRACE_DIR}"
