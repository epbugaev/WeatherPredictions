#!/bin/bash
#SBATCH --job-name=physics-baseline
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# =============================================================================
# Физический бейзлайн на ERA5 1.4° глобус 2000-2010.
#
# Прогоняет матрицу конфигов чистой физики (utils.physics.PurePDEKernel),
# без обучаемых слоёв, и сохраняет метрики (RMSE/MAE/PSNR + физическая
# консистентность) под checkpoints/physics_baseline/<tag>/metrics.parquet.
#
# Submit:
#   bash sh_files/remote_submit.sh sh_files/physics_baseline.sh
#
# Переменные окружения (опциональные):
#   MEMMAP_PATH     — путь к ERA5 1.4° memmap (default: globe_2000_2018.dat)
#   MEAN_STD_PATH   — per-channel mean/std (.npy / .json)
#   START_TIME      — 'YYYY-MM-DD HH:MM:SS' (default: 2000-01-01 00:00:00)
#   END_TIME        — то же (default: 2010-12-31 23:00:00)
#   STRIDE_HOURS    — расстояние между snapshot’ами (default: 24)
#   LEAD_HOURS      — горизонт прогноза (default: 1)
#   BATCH_SIZE      — (default: 4)
#   N_SUBSTEPS      — substeps Euler (default: 12 → lead 1h при dt=300s)
#   BLOCK_DT        — длина substep’а в секундах (default: 300)
#   MATRIX          — 'full' (6 конфигов) | 'fd4_only' | 'weno5_only' | один тэг
#                     (например 'fd4_spherical')
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/ebugaev/WeatherPredictions}}"
cd "${REPO_ROOT}"

: "${MEMMAP_PATH:?Set MEMMAP_PATH to a packed globe memmap .dat before running this checker.}"
MEAN_STD_PATH="${MEAN_STD_PATH:-/home/fratnikov/weather_bench/1.40625deg/mean_std.npy}"
START_TIME="${START_TIME:-2000-01-01 00:00:00}"
END_TIME="${END_TIME:-2010-12-31 23:00:00}"
STRIDE_HOURS="${STRIDE_HOURS:-24}"
LEAD_HOURS="${LEAD_HOURS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_SUBSTEPS="${N_SUBSTEPS:-12}"
BLOCK_DT="${BLOCK_DT:-300}"
MATRIX="${MATRIX:-full}"

echo "[physics_baseline] JobID=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[physics_baseline] git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[physics_baseline] memmap=${MEMMAP_PATH}"
echo "[physics_baseline] window=${START_TIME} .. ${END_TIME}"
echo "[physics_baseline] stride=${STRIDE_HOURS}h, lead=${LEAD_HOURS}h, n_substeps=${N_SUBSTEPS}, block_dt=${BLOCK_DT}s"
echo "[physics_baseline] matrix=${MATRIX}"

# Конфиги для матрицы. tag → "stencil coriolis time_scheme [extra_flags]"
declare -A configs
configs["fd4_constant_euler"]="fd4 constant euler"
configs["fd4_spherical_euler"]="fd4 spherical euler"
configs["fd4_spherical_rk4"]="fd4 spherical rk4"
configs["weno5_constant_euler"]="weno5 constant euler"
configs["weno5_beta_plane_euler"]="weno5 beta_plane euler"
configs["weno5_spherical_euler"]="weno5 spherical euler"

# Выбор подмножества матрицы
declare -a selected_tags
case "${MATRIX}" in
  full)
    selected_tags=(fd4_constant_euler fd4_spherical_euler fd4_spherical_rk4 weno5_constant_euler weno5_beta_plane_euler weno5_spherical_euler)
    ;;
  fd4_only)
    selected_tags=(fd4_constant_euler fd4_spherical_euler fd4_spherical_rk4)
    ;;
  weno5_only)
    selected_tags=(weno5_constant_euler weno5_beta_plane_euler weno5_spherical_euler)
    ;;
  *)
    if [[ -n "${configs[${MATRIX}]:-}" ]]; then
      selected_tags=("${MATRIX}")
    else
      echo "[physics_baseline] Unknown MATRIX=${MATRIX}; valid: full|fd4_only|weno5_only|<tag>" >&2
      exit 1
    fi
    ;;
esac

mkdir -p checkpoints/physics_baseline logs

for tag in "${selected_tags[@]}"; do
  read -r stencil coriolis time_scheme <<<"${configs[$tag]}"
  out_dir="checkpoints/physics_baseline/${tag}"
  echo ""
  echo "=========================================================================="
  echo "[physics_baseline] running tag=${tag}  stencil=${stencil}  coriolis=${coriolis}  time=${time_scheme}"
  echo "=========================================================================="
  python tools/physics_baseline.py \
    --memmap-path "${MEMMAP_PATH}" \
    --mean-std-path "${MEAN_STD_PATH}" \
    --start-time "${START_TIME}" \
    --end-time "${END_TIME}" \
    --stride-hours "${STRIDE_HOURS}" \
    --lead-hours "${LEAD_HOURS}" \
    --stencil "${stencil}" \
    --coriolis "${coriolis}" \
    --time-scheme "${time_scheme}" \
    --block-dt "${BLOCK_DT}" \
    --n-substeps "${N_SUBSTEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${out_dir}"
done

echo ""
echo "[physics_baseline] All ${#selected_tags[@]} configs done. Outputs under checkpoints/physics_baseline/"
ls -la checkpoints/physics_baseline/
