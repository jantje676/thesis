"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import glob


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)

def find_run_number(args):

    ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name + "_0")
    if not os.path.exists(ckpt_dir):
        return "_0"
    else:

        path = "checkpoints/{}*".format(args.dataset)
        runs = glob.glob(path)
        runs.sort()
        print(runs)


        nr_last_run = runs[-1].split("_")[-1]


        if nr_last_run.isdigit():
            nr_last_run = int(nr_last_run)
            nr_next_run = nr_last_run + 1
        else:
            nr_next_run = 0


    return '_' + str(nr_next_run)
