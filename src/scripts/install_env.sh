#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=install_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:10
#SBATCH --output=install_%A.out


module purge
module load 2022
module load Anaconda3/2022.05

cd ..
cd lib

conda env create -f env.yaml
conda activate cgest_env

echo "Environment created and activated"
