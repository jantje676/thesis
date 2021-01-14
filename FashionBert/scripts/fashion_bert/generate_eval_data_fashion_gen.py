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
import h5py
import random
import string

sys.path.append('../../easytransfer/preprocessors')
sys.path.append('/home/kgoei/thesis/FashionBert/easytransfer/preprocessors')
from tokenization import WordpieceTokenizer

" File to convert fashion-gen data to right format "
def main(args):
    random.seed(17)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Called with args:')
    print(args)
    data_dir = args.data_dir
    filename = args.filename

    # get h5 file
    file_path = data_dir + "/" +filename + "validation.h5"
    f = h5py.File(file_path, 'r')

    # get the ids and the captions from the text file for normal cross-modal training
    captions = get_captions(f, args)

    # retrieve requiered model with correct transfrom
    net, transform = get_model(args, device)

    create_data(captions, net, transform, args, device, f)

def create_data(captions, net, transform, args, device, f):
    vocab = get_vocab("{}/fashionbert_pretrain_model_fin/vocab.txt".format(args.bert_dir))
    tokenizer = WordpieceTokenizer(vocab=vocab)
    # open .txt file
    with open('{}/{}/data_caption_{}.txt'.format(args.bert_dir, args.data_out, args.split), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        count = 0
        for i in range(len(captions)):
            id = captions[i][0]
            caption = captions[i][1].lower()
            image = f["input_image"][i]
            features = get_features( image, net, transform, device, args.batch_size)

            # features = ','.join(map(str, np.ones(131072, dtype=float)))
            image_mask = ','.join(map(str, np.ones(64, dtype=int)))
            segment_ids = ','.join(map(str, np.zeros(64, dtype=int)))

            caption_ids = word2id(vocab, tokenizer, caption)

            input_ids = ','.join(map(str, caption_ids))

            writer.writerow((features, image_mask, input_ids, image_mask, segment_ids, int(0), caption, str(i), str(i), str(i)+"_0"))
            count += 1
            if count % 10 == 0:
                print(count)

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

def get_features(image, net, transform, device, batch_size):
    n_segs = 64


    segments = segment_dresses(image, n_segs, transform)

    stack = []
    for i in range(int(n_segs/batch_size)):
        print(i)
        seg_part = segments[i*batch_size:(i+1)*batch_size].to(device)
        dict = net(seg_part)
        features_seg = dict.detach().to("cpu")
        del dict
        torch.cuda.empty_cache()
        stack.append(features_seg)

    hidden_features = torch.cat(stack, dim=0)
    features = hidden_features.numpy().flatten()
    features = ','.join(map(str, features))
    return features

# get the captions and ids from the caption text file
def get_captions(f, args):
    data_captions = {}
    count = 0
    dset = f["index"]
    l_data = dset.shape[0]

    unique_data = []
    prev_caption = ""
    for i in range(l_data):
        if i % 100 == 0:
            print(i)
        try:
            caption = f["input_name"][i][0].decode('UTF-8')
        except:
            continue
        if prev_caption != caption:
            unique_data.append((int(i), caption))
            prev_caption = caption

    random.shuffle(unique_data)
    data_captions = unique_data[:1000]

    print("get_captions: ", len(data_captions))
    return data_captions

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_model(args, device):
    # choose model
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    net.fc = Identity()

    # set to evaluation
    net.eval()

    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

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
    parser.add_argument('--batch_size',help='', default=8, type=int)

    parser.add_argument('--data_dir',help='location data directory', default="../../../data/Fashion_gen/all", type=str)
    parser.add_argument('--bert_dir',help='location data directory', default=".", type=str)
    parser.add_argument('--data_out',help='location of data out', default="eval_img2txt", type=str)
    parser.add_argument('--clothing',help='clothing focus', default="all", type=str)
    parser.add_argument('--split',help='train, val, test', default="train", type=str)
    parser.add_argument('--filename', default="fashiongen_256_256_",help='path to training file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
