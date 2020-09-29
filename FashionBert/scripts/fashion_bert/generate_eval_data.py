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
import math
import pandas as pd
sys.path.append('../../')
from easytransfer.preprocessors.tokenization import WordpieceTokenizer


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Called with args:')
    print(args)

    # get the ids and the captions from the text file for normal cross-modal training
    captions = get_captions(args)

    # retrieve requiered model with correct transfrom
    net, transform = get_model(args, device)

    create_data(captions, net, transform, args, device)

def create_data(captions, net, transform, args, device):
    vocab = get_vocab("fashionbert_pretrain_model_fin/vocab.txt")
    tokenizer = WordpieceTokenizer(vocab=vocab)
    # open .txt file
    with open('{}/data_captions_{}.txt'.format(args.data_out, args.version), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')


        for count, img_id in enumerate(captions.keys()):
            caption = captions[img_id][0]
            path = captions[img_id][1]
            # features = get_features(args.data_dir, path, net, transform, device)
            features = ','.join(map(str, np.ones(6131072, dtype=float)))
            image_mask = ','.join(map(str, np.ones(64, dtype=int)))
            segment_ids = ','.join(map(str, np.zeros(64, dtype=int)))

            caption_ids = word2id(vocab, tokenizer, caption)

            input_ids = ','.join(map(str, caption_ids))

            writer.writerow((features, image_mask, input_ids, image_mask, segment_ids, caption, img_id, img_id, img_id+"_0"))

            if args.early_stop == count:
                break

def word2id(vocab, tokenizer, caption, length=64):
    caption = tokenizer.tokenize(caption)
    caption.insert(0, "[CLS]")
    caption.append("[SEP]")
    ids = np.zeros(64,  dtype=int)
    for i in range(len(caption)):
        ids[i] = vocab[caption[i]]
    return ids

def get_vocab(file):
    vocab = {}

    f = open(file, "r")
    for i,x in enumerate(f):
        vocab[x.strip()] = i
    return vocab

def get_features(data_dir, path, net, transform, device):
    img_path = "{}/{}".format(data_dir, path)
    img = mpimg.imread(img_path)

    segments = segment_dresses(img, 4, transform)

    segments = segments.to(device)
    dict = net(segments)
    hidden_features = dict["out"]

    dim = hidden_features.shape[2]
    pool = nn.AvgPool2d((dim, dim))
    features = pool(hidden_features).to("cpu").squeeze()
    features = features.detach().numpy().flatten()
    features = ','.join(map(str, features))
    return features

# get the captions and ids from the caption text file
def get_captions(args):
    data_captions = {}
    count = 0

    for item in args.list_clothing:
        labels_path = "{}/labels/{}_train_detect_all.txt".format(args.data_dir, item)

        with open(labels_path, newline = '') as file:
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
                data_captions[img_id] = (description, caption[0])
        file.close()

    print("get_captions: ", len(data_captions))
    return data_captions


def get_model(args, device):
    # choose model
    net = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    net = net.backbone

    # set to evaluation
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    net = net.to(device)

    return net, transform

def segment_dresses(im, tiles, transform):
    rows = int(math.sqrt(tiles))
    M = im.shape[0]//rows
    N = im.shape[1]//rows

    # tile images, convert to PIL and transform them directly
    tiles = [transform(Image.fromarray(im[x:x+M,y:y+N])) for x in range(0,rows*M,M) for y in range(0,rows*N,N)]

    stacked_segments = torch.stack(tiles, dim=0)

    return stacked_segments


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--early_stop',help='take lower number of samples for testing purpose', default=None, type=int)
    parser.add_argument('--version',help='add version', default=None, type=str)
    parser.add_argument('--network',help='alex|simCLR|simCLR_pre|layers|sixth|vilbert', default="alex", type=str)
    parser.add_argument('--data_dir',help='location data directory', default="../../../data/Fashion200K", type=str)
    parser.add_argument('--data_out',help='location of data out', default="eval_img2txt", type=str)
    parser.add_argument('--tile', action='store_true', help="use basic tile segmentation")
    parser.add_argument('--clothing',help='clothing item', default="dresses", type=str)
    parser.add_argument("--list_clothing", nargs="+", default=["dresses"])
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
