import h5py
import numpy as np
import argparse

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import csv

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
from segment_dresses import segment_dresses, segment_dresses_tile, segment_dresses_tile_nine


def main(args):
    filename = args.filename
    early_stop = args.early_stop
    data_path = args.data_path
    version = args.version

    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    dset = f["index"]
    l_data = dset.shape[0]

    data_ims_train = create_features(f["input_image"], early_stop)
    data_captions_train = create_captions(f["input_name"], early_stop)
    f.close()


    save_images(data_ims_train, data_path, version, "train")
    save_captions(data_captions_train, data_path, version, "train")

def save_images(ims, data_path, version, split):
    if split == "train":
        filename = "{}/data_ims_{}.npy".format(data_path, version)
    elif split == "test":
        filename = "{}/data_ims_{}_{}.npy".format(data_path, version, split)

    np.save(filename, ims)
    print("Shape data ims {} is: {}".format(split, ims.shape))


def save_captions(captions, data_path, version, split):
    if split =="train":
        filename = '{}/data_captions_{}.txt'.format(data_path, version)
    elif split == "test":
        filename = '{}/data_captions_{}_{}.txt'.format(data_path, version, split)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for caption in captions:
            writer.writerow((caption[0], caption[1]))
    print("length {} captions is {}".format(split, len(captions)))

def create_captions(captions, early_stop):
    cleaned_captions = []
    count = 0
    for caption in captions:
        cleaned_captions.append((count, caption[0].decode("utf-8").lower()))
        count += 1
        if early_stop == count and early_stop != None:
            break
    return cleaned_captions

def create_features(images, early_stop):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net, transform = get_model()

    net = net.to(device)

    features = []
    count = 0
    for image in images:

        # create segments in a dictionary
        segments, bboxes = segment_dresses_tile(image)

        # stack segments to push through net
        stacked_segments = stack_segments(segments, transform)

        feature = get_features(stacked_segments, net, device)
        features.append(feature)

        count += 1

        if count % 10 == 0:
            print(count)
        if early_stop == count and early_stop != None:
            break

    data_out = np.stack(features, axis=0)
    return data_out


def stack_segments(segments, transform):
    segs = []
    for key in segments:
        seg_pil = Image.fromarray(segments[key])

        # transform images
        seg_transformed = transform(seg_pil)

        segs.append(seg_transformed)

    stacked_segments = torch.stack(segs, dim=0)
    return stacked_segments



def get_model():
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

    return net, transform


# segment in seven parts and push through net
def get_features(stacked_segments, net, device ):
    stacked_segments = stacked_segments.to(device)
    feature = net(stacked_segments).to("cpu")
    feature = feature.detach().numpy()
    return feature

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="../../data/Fashion_gen/fashiongen_256_256_train.h5",
                        help='path to training file')
    parser.add_argument('--data_path', default="../../data/Fashion_gen",
                        help='path to data folder.')
    parser.add_argument('--early_stop', default=None, type=int,
                        help='Rank loss margin.')
    parser.add_argument('--version', default="tile",
                        help='version control')
    args = parser.parse_args()
    main(args)
