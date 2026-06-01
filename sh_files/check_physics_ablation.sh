#!/bin/bash
#SBATCH --job-name=phys-abl
#SBATCH --partition=cpu-e-quick
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Forward-абляция стабилизации pure-physics ядра (utils.physics.PurePDEKernel).
# E0 — контроль (взрыв), E1..E5 — кумулятивное добавление стабилизаторов.
# Каждый Ek создаёт свой Comet experiment с уникальным method_name + тегом
# abl_E<k>, так что в Comet UI видны бок о бок. Якорь fixC (production
# scale_diff+detach) прогоняется один раз для верхней планки сравнения.
#
# Все запуски — 48h horizon, 12 IC (jan..dec 2005), block_dt=300s.
#
# Выбор подмножества:  ABL_ONLY="E0 fixC" bash sh_files/check_physics_ablation.sh
# По умолчанию: все E0..E5 + fixC.
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/check_physics_ablation.sh
# локально:
#   bash sh_files/check_physics_ablation.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export PYTHONUNBUFFERED=1

REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

: "${MEMMAP_PATH:?Set MEMMAP_PATH to a packed globe memmap .dat before running this checker.}"
MEAN_STD_PATH="${MEAN_STD_PATH:-}"  # empty = memmap raw (no de-normalisation)
HORIZON="${HORIZON_HOURS:-48}"
BLOCK_DT="${BLOCK_DT_SECONDS:-300}"
YEAR="${YEAR:-2005}"
PYBIN="${PYBIN:-python}"

if [[ -z "${COMET_API_KEY:-}" ]]; then
  echo "[ablation] COMET_API_KEY не задан → все прогоны в --offline режиме."
  OFFLINE_FLAG="--offline"
else
  echo "[ablation] Comet online, workspace=${COMET_WORKSPACE:-<auto>}, project=${COMET_PROJECT_NAME:-WeatherPredictions}"
  OFFLINE_FLAG=""
fi

echo "[ablation] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[ablation] memmap=${MEMMAP_PATH} horizon=${HORIZON}h block_dt=${BLOCK_DT}s year=${YEAR}"

COMMON=(
  --memmap-path "${MEMMAP_PATH}"
  --mean-std-path "${MEAN_STD_PATH}"
  --horizon-hours "${HORIZON}"
  --block-dt-seconds "${BLOCK_DT}"
  --year "${YEAR}"
)

# Базовый интегратор абляции — FD-4 + spherical Coriolis (boundary как у
# new_kernel-дефолтов). Стабилизаторы добавляются КУМУЛЯТИВНО на каждом Ek.
BASE=(
  --stencil fd4 --coriolis spherical
  --boundary-x periodic --boundary-y replicate --boundary-z replicate
)

# label → накопленный набор флагов (E_k включает всё из E_{k-1}).
# E1..E5-флаги появляются в CLI по мере реализации соответствующих шагов;
# до реализации запускайте `ABL_ONLY="E0 fixC"`.
declare -A ABL
ABL[E0]="--time-scheme euler"
ABL[E1]="--time-scheme ssp_rk3"
ABL[E2]="--time-scheme ssp_rk3 --hyperdiffusion --hyperdiff-tau-hours 6"
ABL[E3]="--time-scheme ssp_rk3 --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --polar-filter-lat-deg 60"
ABL[E4]="--time-scheme ssp_rk3 --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --polar-filter-lat-deg 60 --balance-ic dfi --balance-span-hours 1"
ABL[E5]="--time-scheme ssp_rk3 --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --polar-filter-lat-deg 60 --balance-ic dfi --balance-span-hours 1 --advection-form flux --w-diagnostic mass_consistent"
ABL[E6]="--time-scheme semi_implicit --hyperdiffusion --hyperdiff-tau-hours 6 --polar-filter --polar-filter-lat-deg 60 --balance-ic dfi --balance-span-hours 1 --advection-form flux --w-diagnostic mass_consistent"

ABL_ONLY="${ABL_ONLY:-E0 E1 E2 E3 E4 E5 E6 fixC}"

FAILED=()

run_checker () {
  local label="$1"; shift
  echo ""
  echo "============================================================================"
  echo "[$label] starting at $(date '+%F %T')"
  echo "============================================================================"
  if "${PYBIN}" "$@" ${OFFLINE_FLAG}; then
    echo "[$label] OK"
  else
    echo "[$label] FAILED (rc=$?)"
    FAILED+=("$label")
  fi
}

for label in ${ABL_ONLY}; do
  if [[ "${label}" == "fixC" ]]; then
    # Якорь: production scale_diff + .detach() (единственный, кто доживает 48h).
    run_checker "anchor fixC" \
      tools/check_physics_fix.py "${COMMON[@]}" --fix-mode C
    continue
  fi
  flags="${ABL[${label}]:-}"
  if [[ -z "${flags}" ]]; then
    echo "[ablation] неизвестный label '${label}', пропуск"
    continue
  fi
  # shellcheck disable=SC2086
  run_checker "${label} (${flags})" \
    tools/check_physics_new_kernel.py "${COMMON[@]}" "${BASE[@]}" ${flags} \
    --abl-label "${label}"
done

echo ""
echo "============================================================================"
echo "[ablation] DONE at $(date '+%F %T')"
echo "============================================================================"
if (( ${#FAILED[@]} > 0 )); then
  echo "[ablation] FAILED: ${FAILED[*]}"
  exit 1
fi
echo "[ablation] All requested ablations completed."
