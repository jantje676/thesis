#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:15:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1

module load 2019
module load Miniconda2/4.5.12
cp -r $HOME/ $TMPDIR
cp -r $HOME/thesis $TMPDIR

source ~/.bashrc

conda activate thesis

srun python3 /$TMPDIR/thesis/SCAN/train.py

conda deactivate
