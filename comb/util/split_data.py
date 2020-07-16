import numpy as np
from math import floor
import random
import argparse


def main(args):
    clothing = args.clothing
    VERSION_in = args.version
    VERSION_out = args.version
    data_path = args.data_path

    random.seed(args.seed)
    nr_test = args.nr_test
    nr_dev = args.nr_dev


    file = open("{}/{}/data_captions_{}.txt".format(data_path, clothing, VERSION_in),"r")
    captions = file.readlines()
    file.close()

    features = np.load( "{}/{}/data_ims_{}.npy".format(data_path, clothing, VERSION_in))
    caption_count = len(captions)

    rnd_train, rnd_dev, rnd_test = get_dev_test_indices(caption_count, nr_dev, nr_test)

    if features.shape[0] != caption_count:
        print("Shapes do not match!")
        exit()

    if len(rnd_train) + len(rnd_dev) + len(rnd_test) != caption_count:
        print("Lenghts do not match!")
        exit()

    with open('{}/{}/data_captions_{}_train.txt'.format(data_path, clothing, VERSION_out), 'w', newline='') as file:
        for id in rnd_train:
            file.write(captions[id])


    with open('{}/{}/data_captions_{}_dev.txt'.format(data_path, clothing, VERSION_out), 'w', newline='') as file:
        for id in rnd_dev:
            file.write(captions[id])

    with open('{}/{}/data_captions_{}_test.txt'.format(data_path, clothing, VERSION_out), 'w', newline='') as file:
        for id in rnd_test:
            file.write(captions[id])



    features_train = np.stack([features[id] for id in rnd_train], axis=0)
    features_dev = np.stack([features[id] for id in rnd_dev], axis=0)
    features_test = np.stack([features[id] for id in rnd_test], axis=0)

    print("Sizes of different data sets")
    print("train: ", features_train.shape)
    print("dev :", features_dev.shape)
    print("test :", features_test.shape)

    np.save( "{}/{}/data_ims_{}_train.npy".format(data_path, clothing, VERSION_out), features_train)
    np.save( "{}/{}/data_ims_{}_dev.npy".format(data_path, clothing, VERSION_out), features_dev)
    np.save( "{}/{}/data_ims_{}_test.npy".format(data_path, clothing, VERSION_out), features_test)


# shuffle the indices and split into two sets
def get_dev_test_indices(caption_count, dev_size, test_size):
    unique_indices = list(range(0, caption_count))
    random.shuffle(unique_indices)

    if dev_size + test_size > caption_count:
        print("Dev and test are too big for total number of captions {}".format(caption_count))
        exit()

    train_size = caption_count - dev_size - test_size

    rnd_train = unique_indices[:train_size]
    rnd_dev = unique_indices[train_size: train_size + dev_size]
    rnd_test = unique_indices[train_size + dev_size:]

    return rnd_train, rnd_dev, rnd_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=None, help='version control')
    parser.add_argument('--data_path', default="../../data/Fashion200K", help='path to data folder.')
    parser.add_argument('--seed', default=17, type=int, help='seed')
    parser.add_argument('--clothing', default="dresses", type=str, help='clothing item')
    parser.add_argument('--nr_test', default=1000, type=int, help='size of test set')
    parser.add_argument('--nr_dev', default=1000, type=int, help='size of dev set')
    args = parser.parse_args()
    main(args)
