#!/bin/bash
#SBATCH --job-name=ablation-gradguard-1gpu
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --constraint="type_e|type_f"
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Real-ERA5 before/after ablation of the HybridBlock gradient guard.
# Stages the packed USA memmap to node-local storage, provisions (idempotently)
# a detached worktree of the pre-fix commit 4d2dea8, and runs
# tools/exp_gradguard_ablation.py once per condition on the SAME staged data.
set -euo pipefail
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

_sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

BEFORE_COMMIT="${BEFORE_COMMIT:-4d2dea8}"
BEFORE_WT="${BEFORE_WT:-${HOME}/wt_gg_before_${BEFORE_COMMIT}}"
if [[ ! -f "${BEFORE_WT}/train.py" ]]; then
  echo "[ablation] provisioning pre-fix worktree ${BEFORE_WT} @ ${BEFORE_COMMIT}"
  git -C "${REPO_ROOT}" worktree add --detach "${BEFORE_WT}" "${BEFORE_COMMIT}"
fi

ORIG_MEMMAP="${ORIG_MEMMAP:-${WEATHERPRED_USA_MEMMAP:-}}"
: "${ORIG_MEMMAP:?Set ORIG_MEMMAP or WEATHERPRED_USA_MEMMAP to the packed USA memmap .dat for v4.}"
STAGE_DIR="${STAGE_DIR:-/tmp/${USER:-$(id -un)}/era5_stage_${SLURM_JOB_ID:-local}}"
mkdir -p "${STAGE_DIR}"
echo "[ablation] staging memmap ${ORIG_MEMMAP} -> ${STAGE_DIR}/"
cp "${ORIG_MEMMAP}" "${STAGE_DIR}/"
weatherpred__orig_meta="${ORIG_MEMMAP%.dat}.meta.json"
if [[ -f "${weatherpred__orig_meta}" ]]; then
  cp "${weatherpred__orig_meta}" "${STAGE_DIR}/"
fi
STAGED_MEMMAP="${STAGE_DIR}/$(basename "${ORIG_MEMMAP}")"

trap 'rm -rf "${STAGE_DIR}" 2>/dev/null || true' EXIT

JOB="${SLURM_JOB_ID:-local}"
echo "[ablation] === AFTER (post-fix, ${REPO_ROOT}) ==="
python "${REPO_ROOT}/tools/exp_gradguard_ablation.py" \
  --repo "${REPO_ROOT}" --label after --memmap "${STAGED_MEMMAP}" \
  --arms fixedeq massconsistent legacy_hybrid --steps 12 \
  --out "${REPO_ROOT}/logs/gg_after_${JOB}.json"
echo "[ablation] === BEFORE (pre-fix, ${BEFORE_WT} @ ${BEFORE_COMMIT}) ==="
python "${REPO_ROOT}/tools/exp_gradguard_ablation.py" \
  --repo "${BEFORE_WT}" --label before --memmap "${STAGED_MEMMAP}" \
  --arms fixedeq massconsistent legacy_hybrid --steps 12 \
  --out "${REPO_ROOT}/logs/gg_before_${JOB}.json"
echo "[ablation] done: logs/gg_{before,after}_${JOB}.json"
