#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1

module load 2019
module load Miniconda2/4.5.12
cp -r $HOME/ $TMPDIR

source ~/.bashrc

conda activate thesis

srun python3 $TMPDIR/thesis/Beta-VAE/main.py --dataset dsprites --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --viz_name dsprites_B_gamma100_z10

conda deactivate
