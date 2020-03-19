#! /bin/sh




python main.py --dataset Fashion200K --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 5 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --viz_name Fashion200K --viz_on False \
    --display_step 20 --dset_dir ../data --save_step 4 --resize padding --ratio_width 128 \
    --gather_step 1 --display_step 1 --image_size 64
