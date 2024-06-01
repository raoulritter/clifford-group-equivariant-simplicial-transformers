#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=our_gatr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:10:10
#SBATCH --output=fin_model_%A.out


module purge
module load 2022
module load Anaconda3/2022.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate cgest_env

cd ..

srun python lib/main.py
