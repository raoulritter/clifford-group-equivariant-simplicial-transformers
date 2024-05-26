#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=hpn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:10:10
#SBATCH --output=hp_trans_nodes_%A.out


module purge
module load 2022
module load Anaconda3/2022.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate dl2023
cd ..
srun python hyperparameter_testing.py --num_edges 0
