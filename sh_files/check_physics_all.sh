#!/bin/bash
#SBATCH --job-name=phys-all
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Прогон всех физических чекеров (старые + новые) подряд на одной CPU-ноде.
# Каждый чекер создаёт свой Comet experiment с уникальным method_name, так что
# в Comet UI методы видны бок о бок. Aggregator пропагирует NaN между IC, чтобы
# метрики между методами никогда не были идентичными при наличии нестабильности.
#
# Все запуски — 48h horizon, 12 IC (jan..dec 2005), block_dt=300s.
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/check_physics_all.sh
# или локально:
#   bash sh_files/check_physics_all.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

MEMMAP_PATH="${MEMMAP_PATH:-/home/ebugaev/era5_memmap/predformer_globe_2000_2018.dat}"
MEAN_STD_PATH="${MEAN_STD_PATH:-}"  # empty = memmap raw (no de-normalisation)
HORIZON="${HORIZON_HOURS:-48}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"

# Comet creds — задайте через .env или окружение перед sbatch.
#   export COMET_API_KEY=...
#   export COMET_WORKSPACE=...
#   export COMET_PROJECT_NAME=WeatherPredictions
# Если COMET_API_KEY не задан, каждый чекер запишется в OfflineExperiment
# (logs/comet_offline/<method_name>/*.zip), синхронизировать руками через
# `comet upload <zip>`.

if [[ -z "${COMET_API_KEY:-}" ]]; then
  echo "[check_physics_all] COMET_API_KEY не задан → запускаем все чекеры в --offline режиме."
  OFFLINE_FLAG="--offline"
else
  echo "[check_physics_all] Comet online ML, workspace=${COMET_WORKSPACE:-<auto>}, project=${COMET_PROJECT_NAME:-WeatherPredictions}"
  OFFLINE_FLAG=""
fi

echo "[check_physics_all] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[check_physics_all] memmap=${MEMMAP_PATH}"
echo "[check_physics_all] horizon=${HORIZON}h, block_dt=${BLOCK_DT}s, year=${YEAR}"

COMMON=(
  --memmap-path "${MEMMAP_PATH}"
  --mean-std-path "${MEAN_STD_PATH}"
  --horizon-hours "${HORIZON}"
  --block-dt-seconds "${BLOCK_DT}"
  --year "${YEAR}"
)

# Накопить failed methods, чтобы один blow-up не клал всю серию.
FAILED=()

run_checker () {
  local label="$1"; shift
  echo ""
  echo "============================================================================"
  echo "[$label] starting at $(date '+%F %T')"
  echo "============================================================================"
  if python "$@" ${OFFLINE_FLAG}; then
    echo "[$label] OK"
  else
    echo "[$label] FAILED (rc=$?)"
    FAILED+=("$label")
  fi
}

# Старые методы (через utils.old_physics — байт-в-байт legacy paper).
run_checker "weathergft (FD-4 + Euler + const)" \
  tools/check_physics_weathergft.py "${COMMON[@]}"

run_checker "predformergft (WENO-5 + Euler + beta-plane + AMR)" \
  tools/check_physics_predformergft.py "${COMMON[@]}"

run_checker "weathergft_3 (FD-4 + RK4 + spherical + turb + rad + Kuo)" \
  tools/check_physics_weathergft_3.py "${COMMON[@]}"

# Новые методы (через utils.physics.PurePDEKernel — adiabatic_omega + R_d).
run_checker "new_kernel fd4 + Euler + spherical" \
  tools/check_physics_new_kernel.py "${COMMON[@]}" \
  --stencil fd4 --time-scheme euler --coriolis spherical \
  --boundary-x periodic --boundary-y replicate --boundary-z replicate

run_checker "new_kernel fd4 + RK4 + spherical" \
  tools/check_physics_new_kernel.py "${COMMON[@]}" \
  --stencil fd4 --time-scheme rk4 --coriolis spherical \
  --boundary-x periodic --boundary-y replicate --boundary-z replicate

run_checker "new_kernel weno5 + Euler + beta_plane" \
  tools/check_physics_new_kernel.py "${COMMON[@]}" \
  --stencil weno5 --time-scheme euler --coriolis beta_plane \
  --boundary-x reflect --boundary-y replicate --boundary-z replicate

# Diagnostic fix-modes (A/B/C) — для регрессии vs paper-формулы.
for fix in A B C; do
  run_checker "check_physics_fix mode=${fix}" \
    tools/check_physics_fix.py "${COMMON[@]}" --fix-mode "${fix}"
done

echo ""
echo "============================================================================"
echo "[check_physics_all] DONE at $(date '+%F %T')"
echo "============================================================================"
if (( ${#FAILED[@]} > 0 )); then
  echo "[check_physics_all] FAILED methods: ${FAILED[*]}"
  exit 1
fi
echo "[check_physics_all] All methods completed."
