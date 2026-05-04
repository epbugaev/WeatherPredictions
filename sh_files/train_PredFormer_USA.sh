#!/bin/bash
#SBATCH --job-name=PredFormer-USA
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
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/ebugaev:$PYTHONPATH

module load Python/Anaconda_v11.2021

source deactivate
source activate /home/ebugaev/.conda/envs/weatherpred-gft-fix

cd /home/ebugaev/WeatherPredictions

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi
python - <<'PY'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("torch.cuda.devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

srun python /home/ebugaev/WeatherPredictions/train.py \
  --config configs/predformer_usa.yaml \
  --nodes 1 \
  --gpus_per_node 1
