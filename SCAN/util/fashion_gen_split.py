import numpy as np
from math import floor
import random
import argparse


"""
Script to split the devtest data in two seperate files. The test data already has its own file.
The data is filtered for unique ideas, because there are multiple pictures with every image
"""

def main(args):
    VERSION = args.version
    data_path = args.data_path

    seed = args.seed
    dev_size = args.dev_size
    test_size = args.test_size
    total = dev_size + test_size
    random.seed(seed)

    # read captions
    file = open("{}/data_captions_{}_devtest.txt".format(data_path, VERSION),"r")
    captions = file.readlines()
    file.close()

    # read features
    features = np.load("{}/data_ims_{}_devtest.npy".format(data_path, VERSION))


    # create list with unique indices
    unique_indices = get_unique_indices(captions)
    rnd_dev, rnd_test = get_dev_test_indices(unique_indices, dev_size, test_size)

    features_dev = np.stack([features[id] for id in rnd_dev], axis=0)
    features_test = np.stack([features[id] for id in rnd_test], axis=0)
    np.save( "{}/data_ims_{}_dev.npy".format(data_path, VERSION), features_dev)
    np.save( "{}/data_ims_{}_test.npy".format(data_path, VERSION), features_test)


    with open("{}/data_captions_{}_dev.txt".format(data_path, VERSION), 'w', newline='') as file:
        for id in rnd_dev:
            file.write(captions[id])

    with open("{}/data_captions_{}_test.txt".format(data_path, VERSION), 'w', newline='') as file:
        for id in rnd_test:
            file.write(captions[id])

# shuffle the indices and split into two sets
def get_dev_test_indices(unique_indices, dev_size, test_size):
    random.shuffle(unique_indices)

    total = dev_size + test_size
    if total < len(unique_indices):
        rnd_dev = unique_indices[:dev_size]
        rnd_test = unique_indices[dev_size:]
    else:
        dev_size = floor(len(unique_indices) * (dev_size/total))
        rnd_dev = unique_indices[:dev_size]
        rnd_test = unique_indices[dev_size:]
    return rnd_dev, rnd_test

# filter the captions so only one image per caption is left behind
def get_unique_indices(captions):
    unique_indices = []
    prev_caption = ""
    for line in captions:
        try:
            id, caption = line.split("\t")
        except:
            continue
        if prev_caption != caption:
            unique_indices.append(int(id))
            prev_caption = caption

    return unique_indices



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=None, help='version control')
    parser.add_argument('--data_path', default="../../data/Fashion_gen/all", help='path to data folder.')
    parser.add_argument('--seed', default=17, type=int, help='seed')
    parser.add_argument('--dev_size', default=1000, type=int, help='size of dev set')
    parser.add_argument('--test_size', default=100, type=int, help='size of test set')
    args = parser.parse_args()
    main(args)
