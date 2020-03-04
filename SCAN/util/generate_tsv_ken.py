

import glob
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
import random
import json
import torchvision.models as models
import matplotlib.image as mpimg
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from math import floor
import torch.nn as nn


def segment(img, net, img_id, transform):


    H, W, C = img.shape
    segments = {}
    bboxes = []

    segments["top"] = img[: floor(0.35*H) , : , :]
    bboxes.append((0,0,20,20))
    segments["full_skirt"] = img[floor(0.35*H):  , : , :]
    bboxes.append((0,0,20,20))
    segments["skirt_above_knee"] = img[floor(0.35*H): floor(0.75*H) , : , :]
    bboxes.append((0,0,20,20))
    segments["neckline"] = img[: floor(0.2*H) , : , :]
    bboxes.append((0,0,20,20))
    segments["left_sleeve"] = img[: floor(0.5*H) , : floor(0.5*W) , :]
    bboxes.append((0,0,20,20))
    segments["right_sleeve"] = img[: floor(0.5*H) , floor(0.5*W): , :]
    bboxes.append((0,0,20,20))
    segments["full"] = img
    bboxes.append((0,0,20,20))

    features = []
    for key in segments:
        seg_pil = Image.fromarray(segments[key])
        seg_transformed = transform(seg_pil)

        seg_transformed = seg_transformed.unsqueeze(0)
        feature = net(seg_transformed)
        feature = feature.squeeze()
        features.append(feature)
        # push trough net
        # take features
        # append to
    return {
        "image_id": img_id,
        "image_h": H,
        "image_w": W,
        "num_boxes": 7,
        "boxes" :bboxes,
        "features": features

    }

def load_image_ids():
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    path = "../../data/Fashion200K/test_no_seg/**/*.jpeg"
    img_paths = glob.glob(path)
    for img_path in img_paths:
        image_id = img_path.split("/")[5]
        split.append((img_path, image_id))
    return split

def generate_tsv(image_ids):
    net = models.alexnet(pretrained=True)

    net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])
    net.eval()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    for img_id in image_ids:
        img_path = img_id[0]
        img_idx = img_id[1]

        img=mpimg.imread(img_path)
        hello = segment(img, net, img_id, transform)


        break
        # write to file
    return


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)



    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    image_ids = load_image_ids()
    print(image_ids)
    generate_tsv(image_ids)
