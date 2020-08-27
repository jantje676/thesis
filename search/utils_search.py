import numpy as np

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def norm(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def norm_t2i(x):
    sorted = np.unique(x)
    x_max = sorted[-1]


    if sorted[0] == np.NINF and len(sorted) != 1 :
        x_min = sorted[1]
    else:
        x_min = sorted[0]

    # calculate norm for all elements that not contain a np.NINF
    x_norm = np.where(x != np.NINF,(x - x_min) / (x_max - x_min), x )
    return x_norm
