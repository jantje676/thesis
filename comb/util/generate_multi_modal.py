import glob
import argparse
import time, os, sys
import base64
import numpy as np
import cv2
import csv
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch
# sys.path.append('../../')
# sys.path.append("/$HOME/thesis/comb")
from segment_dresses import segment_dresses, segment_dresses_tile, segment_dresses_tile_nine
from generate_tsv_ken import get_model, get_features
from layers_model import LayersModel

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Called with args:')
    print(args)

    # retrieve requiered model with correct transfrom
    net, transform = get_model(args, device)

    # generate data
    generate_data(args.data_dir, args.data_out, args.tile, args.network, args.version, net, device, transform)


def generate_data(data_dir, data_out, tile, network, version, net, device, transform):
    count_stop = 0
    data = []
    data_path = "{}/Fashion200K_multi/dresses/data_captions_laenen_1k_test.txt".format(data_dir)
    with open(data_path, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')

        for caption in caption_reader:
            img_path = caption[0]
            img_adress = "{}/Fashion200K/{}".format(data_dir, img_path)
            img = mpimg.imread(img_adress)

            # segment dresses and retreive segmentations
            if args.tile:
                segments, bboxes = segment_dresses_tile(img)
            else:
                segments, bboxes = segment_dresses(img)

            # create features from segmentations
            seg = get_features(img, net, 1, transform, segments, bboxes, device, network)

            temp = np.frombuffer( base64.b64decode(seg["features"]), dtype=np.float32)
            temp = temp.reshape((seg["num_boxes"],-1))

            count_stop += 1
            if count_stop % 10 == 0:
                print(count_stop)

            data.append(temp)

    data_out = np.stack(data, axis=0)
    # save images
    np.save( "{}/data_ims_{}_test.npy".format(args.data_out, args.version), data_out)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--version',help='add version', default=None, type=str)
    parser.add_argument('--network',help='alex|simCLR|simCLR_pre|layers|sixth', default="alex", type=str)
    parser.add_argument('--data_dir',help='location data directory', default="../../data", type=str)
    parser.add_argument('--data_out',help='location of data out', default="../../data/Fashion200K_multi/dresses", type=str)
    parser.add_argument('--tile', action='store_true', help="use basic tile segmentation")
    parser.add_argument('--clothing',help='clothing item', default="dresses", type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
