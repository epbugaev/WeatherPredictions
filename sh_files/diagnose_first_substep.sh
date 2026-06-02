#!/bin/bash
#SBATCH --job-name=phys-diag-substep
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO_ROOT_FOR_CONTRACT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${HOME}/WeatherPredictions}}"
_sc_here="${REPO_ROOT_FOR_CONTRACT}/sh_files"
source "${_sc_here}/_shell_contract.sh" "${_sc_here}"
python tools/diagnose_first_substep.py
