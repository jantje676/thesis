#! /bin/sh

python main.py --dataset dresses --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1000 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --viz_name dresses --viz_on False
