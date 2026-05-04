#!/bin/bash
#SBATCH --job-name=SimVP-USA
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-18:00:00
#SBATCH --mail-user=egor07072003@gmail.com
#SBATCH --mail-type=ALL

export NGPUS=1
export OMP_NUM_THREADS=4
export PYTHONPATH=/home/ebugaev:$PYTHONPATH

module load Python/Anaconda_v11.2021

source deactivate
source activate /home/ebugaev/.conda/envs/weatherpred-gft-fix

cd /home/ebugaev/WeatherPredictions

srun torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_port=25683 \
  /home/ebugaev/WeatherPredictions/train.py \
  --config configs/simvp_usa.yaml \
  --nodes 1 \
  --gpus_per_node 1
