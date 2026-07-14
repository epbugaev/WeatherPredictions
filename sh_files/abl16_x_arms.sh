#!/usr/bin/env bash
#SBATCH --job-name=abl16L-x
#SBATCH --account=proj_1715
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
# =============================================================================
# exp16, армы X1/X2 — недостающая клетка факторного плана 2×2 (уравнения × Q_θ).
#
#   X1 = R0 + Q_θ                 — Q_θ БЕЗ уравнений. Вместе с R0/R3a/R3 замыкает
#                                   план и даёт член взаимодействия:
#                                   (R3 − R3a) − (X1 − R0). См. docs/architecture.md §7.
#   X2 = X1 с обнулённой гео      — Q_θ единственная голова, читающая орографию,
#                                   широту и маску суши; X1 − X2 = вклад географии.
#
# Ни в одном арме нет физики ⇒ скорость как у R0: 17:12 на a100 (type_e) против ~28 ч
# у физ-армов. Walltime 24 ч с запасом; constraint отсекает v100 (там в 6–9× дольше).
#
# Чекпоинты пишутся туда же, где лежат остальные армы лестницы, чтобы харнесс метрик
# (docs/experiments/16_model_ablation_ladder/metrics/) нашёл их без правок.
#
# Запуск:
#   cd ~/wt_fix_v2 && mkdir -p logs && sbatch sh_files/abl16_x_arms.sh
# =============================================================================
set -euo pipefail

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export WEATHERPRED_CONDA_ENV_NAME=pi-iamvp
export MEMMAP_PATH_OVERRIDE="${HOME}/era5_memmap/predformer_usa_2000_2004.dat"
export MEMMAP_META_PATH_OVERRIDE="${HOME}/era5_memmap/predformer_usa_2000_2004.meta.json"
export WEATHERPRED_CHECKPOINT_BASE="${HOME}/abl16_long_ckpt"
export NNODES=1 NGPUS=1

REPO_ROOT="${REPO_ROOT:-${HOME}/wt_fix_v2}"
export REPO_ROOT

ARMS=(
  abl16_x1_qtheta_no_eq_t12
  abl16_x2_qtheta_no_geo_t12
)
arm="${ARMS[${SLURM_ARRAY_TASK_ID}]}"

echo "[abl16L-x] arm=${arm} node=$(hostname) repo=${REPO_ROOT} start=$(date +%H:%M:%S)"
bash "${REPO_ROOT}/sh_files/launch_train.sh" "${REPO_ROOT}/configs/abl16_long/${arm}.yaml"
echo "[abl16L-x] done arm=${arm} end=$(date +%H:%M:%S)"
