#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1



#module purge
#module load eb

module load 2019
module load Python/3.7.5-foss-2018b
module load cuDNN/7.6.3-CUDA-10.0.130
module load NCCL/2.4.7-CUDA-10.0.130

srun python3 $HOME/glow/train.py --num_epochs=50