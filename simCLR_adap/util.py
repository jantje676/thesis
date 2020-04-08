import argparse
import glob


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def find_run_name(opt):
    path = "{}run*".format(opt.output_dir)
    runs = glob.glob(path)
    runs.sort()
    if len(runs) == 0:
        return 0
    elif len(runs) > 0:
        nr_next_run = len(runs)

    return nr_next_run
