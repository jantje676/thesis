

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
import sys
import torch
sys.path.append('../../simclr')
sys.path.append('../../SimCLR_pre')
from models.resnet_simclr import ResNetSimCLR
from modules.simclr import SimCLR

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']



class padd(object):
    def __call__(self,img):
        W, H = img.size
        # check if image is rectangle shaped
        if H > W:
            diff = H - W
            desired_size = H
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(img, (diff//2, 0))
        elif W > H:
            diff = W - H
            desired_size = W
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(img, (0, diff//2))
        elif W == H:
            new_im = img
        return  new_im


# get the captions and ids from the caption text file
def get_captions():
    data_captions = {}
    image_ids_captions =[]
    count = 0

    with open(args.labels, newline = '') as file:
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

        if args.network == "alex":
            feature = feature.squeeze()
        elif args.network == "simCLR" or args.network == "simCLR_pre":
            feature = feature[0].squeeze()
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
    path = args.image_path

    img_paths = glob.glob(path)
    for img_path in img_paths:
        image_id = img_path.split("_")[-2]
        image_id = image_id.split("/")[-1]
        split.append((img_path, image_id))

    return split


def generate_tsv(image_ids, args):
    data = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.network == "alex":
        # choose model
        net = models.alexnet(pretrained=True)
        # take aways the last layers
        net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])

        # set to evaluation
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "simCLR":
        net = ResNetSimCLR("resnet18", args.output_dim)
        net.load_state_dict(torch.load("../../simCLR/runs_simCLR/run_26/checkpoints/model.pth", map_location=torch.device(device)))
        net.eval()
        # to transformations here
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(args.input_shape_height, args.input_shape_width)),
            transforms.ToTensor()])

    elif args.network == "simCLR_pre":
        net = SimCLR(args)
        net.load_state_dict(torch.load(args.checkpoint_simCLR_pre, map_location=torch.device(device)))
        net.eval()

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor()])

    count_stop = 0
    # open file to write data to TSV
    with open("{}/tsv_output_segmentations_{}.tsv".format(args.data_dir, args.version), 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
        print("Started reading images")
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

            if count_stop % 10 == 0:
                print(count_stop)
    tsvfile.close()
    return data


def combine_data_captions(image_ids_images, data, data_captions, image_ids_captions, args):

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
    np.save( "{}/data_ims_{}.npy".format(args.data_dir, args.version), data_out)

    with open('{}/data_captions_{}.txt'.format(args.data_dir, args.version), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        for id in ids_needed:
            writer.writerow((id, data_captions[id]))

    return

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--early_stop',help='take lower number of samples for testing purpose', default=None, type=int)
    parser.add_argument('--version',help='add version', default=None, type=str)
    parser.add_argument('--network',help='alex|simCLR|simCLR_pre', default="alex", type=str)
    parser.add_argument('--labels',help='location of labels', default="../../data/Fashion200K/labels/dress_train_detect_all.txt", type=str)
    parser.add_argument('--image_path',help='location of images', default="../../data/Fashion200K/pictures_only/pictures_only/*.jpeg", type=str)
    parser.add_argument('--checkpoint_simCLR_pre',help='location pretrained model', default="../../SimCLR_pre/checkpoint_100.tar", type=str)
    parser.add_argument('--data_dir',help='location data directory', default="../../data/Fashion200K", type=str)


    # WHEN PRETRAINED simCLR IS USED
    parser.add_argument('--output_dim',help='if simCLR is used, size of output dim', default=256, type=int)
    parser.add_argument('--input_shape_width', default=96, type=int, help='(W, H, C) for when pretrained network is used' )
    parser.add_argument('--input_shape_height', default=192, type=int, help='(W, H, C) for when pretrained network is used')

    # WHEN simCLR_pre IS USED
    parser.add_argument('--resnet',help='which resnet to use', default="resnet50", type=str)
    parser.add_argument('--normalize',help='use normalize', default="True", type=str2bool)
    parser.add_argument('--projection_dim',help='size of projection dim', default=64, type=int)

    args = parser.parse_args()

    return args



def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    combine_data_captions(image_ids_captions, data, data_captions, image_ids_captions, args)
