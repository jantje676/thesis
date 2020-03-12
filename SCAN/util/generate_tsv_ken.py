

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

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# get the captions and ids from the caption text file
def get_captions():
    data_captions = {}
    image_ids_captions =[]
    count = 0
    with open("../../data/Fashion200K/labels/dress_train_detect_all.txt", newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')

        # read every line
        for caption in caption_reader:

            # extract the id
            img_id = caption[0].split("_")[-2]
            img_id = img_id.split("/")[-1]

            # ids have duplicate for every picture
            if img_id in data_captions.keys():
                count += 1
                continue

            # get the description
            description = caption[2]
            data_captions[img_id] = description
            image_ids_captions.append(img_id)
    file.close()
    return data_captions, image_ids_captions

# segment in seven parts and push through net
def segment(img, net, img_idx, transform):
    H, W, C = img.shape
    segments = {}

    # 1=x_1, 2=y_1, 3=x_2, 4 =y_2 linkerbovenhoek=(x_1, y_1) rechteronderhoek=(x_2, y_2)
    bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                      [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)]])

    # segment dress in seven parts according to laenen
    segments["top"] = img[: floor(0.35*H) , : , :]
    segments["full_skirt"] = img[floor(0.35*H):  , : , :]
    segments["skirt_above_knee"] = img[floor(0.35*H): floor(0.75*H) , : , :]
    segments["neckline"] = img[: floor(0.2*H) , : , :]
    segments["left_sleeve"] = img[: floor(0.5*H) , : floor(0.5*W) , :]
    segments["right_sleeve"] = img[: floor(0.5*H) , floor(0.5*W): , :]
    segments["full"] = img

    features = []

    # push segments through the net
    for key in segments:
        seg_pil = Image.fromarray(segments[key])

        # transform images
        seg_transformed = transform(seg_pil)

        # add extra dimension
        seg_transformed = seg_transformed.unsqueeze(0)
        feature = net(seg_transformed)
        feature = feature.squeeze()
        features.append(feature.detach().numpy())

    features = np.stack(features, axis=0)
    return {
        "image_id": int(img_idx),
        "image_h": int(H),
        "image_w": int(W),
        "num_boxes": int(7),
        "boxes" :base64.b64encode(bboxes),
        "features": base64.b64encode(features)
    }

# load the images ids and path from the folder structure
def load_image_ids():

    split = []
    path = "../../data/Fashion200K/pictures_only/pictures_only/*.jpeg"

    img_paths = glob.glob(path)
    for img_path in img_paths:
        image_id = img_path.split("_")[-2]
        image_id = image_id.split("/")[-1]
        split.append((img_path, image_id))

    return split

def generate_tsv(image_ids, args):
    data = {}

    # choose model
    net = models.alexnet(pretrained=True)
    # take aways the last layers
    net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])

    # set to evaluation
    net.eval()

    # transformations for the net
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    count_stop = 0
    # open file to write data to TSV
    with open("../../data/Fashion200K/tsv_output_segmentations.tsv", 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)

        # for every image create seven segmentations and push them through pretrained net
        for img_id in image_ids:
            img_path = img_id[0]
            img_idx = img_id[1]

            img=mpimg.imread(img_path)

            # create features from segmentations
            seg = segment(img, net, img_idx, transform)

            # segment in seven and write
            writer.writerow(seg)

            temp = np.frombuffer( base64.b64decode(seg["features"]), dtype=np.float32)
            temp = temp.reshape((seg["num_boxes"],-1))
            data[img_idx] = temp
            count_stop +=1
            if count_stop == args.early_stop:
                break
    tsvfile.close()
    return data


def combine_data_captions(image_ids_images, data, data_captions, image_ids_captions):

    ids_needed = []
    for id in image_ids_captions:
        if id in data.keys():
            ids_needed.append(id)
    data_out = np.stack([data[id] for id in ids_needed], axis=0)

    print("Shape of data_out is {}".format(data_out.shape))
    print(len(ids_needed))
    if data_out.shape[0] != len(ids_needed):
        print("length should be equal!!")
        exit()
    np.save( "../../data/Fashion200K/data_ims.npy", data_out)


    with open('../../data/Fashion200K/data_captions.txt', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        for id in ids_needed:
            writer.writerow((id, data_captions[id]))

    return

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--early_stop',
                        help='take lower number of samples for testing purpose',
                        default=None, type=int)

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    # get the ids and the captions from the text file
    data_captions, image_ids_captions = get_captions()

    # load the images ids from the image folder stucture, return list with tuples (file_path, image_id)
    image_ids_images = load_image_ids()

    # generate features and write to .tsv
    data = generate_tsv(image_ids_images, args)

    # create numpy stack from features and match with captions
    combine_data_captions(image_ids_captions, data, data_captions, image_ids_captions)
