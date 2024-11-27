#! /bin/bash
#SBATCH --job-name="test convert_input script"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1 
#SBATCH --ntasks=1
#SBATCH --time=3-0:0
#SBATCH --mail-user=egor07072003@gmail.com
#SBATCH --mail-type=ALL
module load Python/Anaconda_v11.2021

source deactivate 
source activate /home/epbugaev/.conda/envs/my_py_env1

# Executable
srun python3 /home/epbugaev/WeatherPredictions/convert_input.py
