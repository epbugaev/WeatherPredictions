#!/usr/bin/env bash
# =============================================================================
# One-shot ERA5 -> memmap repack (§2.5 of the dataloader plan).
#
# Wraps ``python tools/repack_era5.py`` in a SLURM submission. Reads the
# configured year range and spatial cut from env (defaults match
# configs/bench_memmap.yaml: 2000-2004, cut=[75 107 164 228]) and writes
# ``<DST>/<PREFIX>.{dat,meta.json}``.
#
# Usage:
#   sbatch sh_files/repack_era5.sh
#   REPACK_END=2010 sbatch sh_files/repack_era5.sh
#   REPACK_DST=/scratch/era5_memmap REPACK_PREFIX=full_50yr ... sbatch ...
#
# Full globe at 5.625deg, 2000-2018 (~95 GB output):
#   REPACK_PREFIX=predformer_globe_2000_2018 \
#   REPACK_START_YEAR=2000 REPACK_END_YEAR=2018 \
#   REPACK_CUT="0 32 0 64" \
#   REPACK_SRC=/home/fratnikov/weather_bench/5.625deg \
#   REPACK_RES_SUFFIX=5.625deg \
#   sbatch sh_files/repack_era5.sh
# =============================================================================
#SBATCH --partition=rocky
#SBATCH --qos=rocky
#SBATCH --constraint="type_e|type_f"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=repack_era5
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _sc_here="${SLURM_SUBMIT_DIR}/sh_files"
else
  _sc_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
# shellcheck source=sh_files/_shell_contract.sh
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"

DST="${REPACK_DST:-/home/ebugaev/era5_memmap}"
PREFIX="${REPACK_PREFIX:-predformer_usa_2000_2004}"
START_YEAR="${REPACK_START_YEAR:-2000}"
END_YEAR="${REPACK_END_YEAR:-2004}"
CUT="${REPACK_CUT:-75 107 164 228}"
SRC="${REPACK_SRC:-/home/fratnikov/weather_bench/1.40625deg}"
RES_SUFFIX="${REPACK_RES_SUFFIX:-1.40625deg}"

mkdir -p "${DST}"

echo "[repack] src=${SRC} dst=${DST}/${PREFIX} years=${START_YEAR}-${END_YEAR} cut=${CUT} res=${RES_SUFFIX}"
echo "[repack] python=$(command -v python) host=$(hostname)"

python tools/repack_era5.py \
  --src "${SRC}" \
  --dst "${DST}" \
  --prefix "${PREFIX}" \
  --start-year "${START_YEAR}" \
  --end-year "${END_YEAR}" \
  --cut ${CUT} \
  --res-suffix "${RES_SUFFIX}"

echo "[repack] artefacts:"
ls -la "${DST}/${PREFIX}".dat "${DST}/${PREFIX}".meta.json
