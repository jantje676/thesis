import glob
import os
import json
import random

def save_hyperparameters(log_path, opt):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open('{}/commandline_args.txt'.format(log_path), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

def find_run_name(opt):
    path = "runs/run*"
    runs = glob.glob(path)
    runs.sort()
    if len(runs) == 0:
        return opt
    elif len(runs) > 0:
        nr_next_run = len(runs) + 1
    opt.model_name = './runs/run{}/checkpoint'.format(nr_next_run)
    opt.logger_name = './runs/run{}/log'.format(nr_next_run)

    return opt

def get_random_indx(nr_examples, max_len):
    rnd = [x for x in range(max_len)]
    random.shuffle(rnd)
    rnd = rnd[:nr_examples]
    return rnd
