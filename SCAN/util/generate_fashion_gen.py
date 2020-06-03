import h5py
import numpy as np
import argparse
from math import floor

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import csv

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms



def main(args):
    filename = args.filename
    early_stop = args.early_stop
    data_path = args.data_path
    version = args.version
    data_path_out = args.data_path_out

    file_path = data_path + "/" +filename + "train.h5"
    f = h5py.File(file_path, 'r')
    data_ims_train = create_features(f["input_image"], early_stop)
    data_captions_train = create_captions(f["input_name"], early_stop)
    f.close()

    save_images(data_ims_train, data_path_out, version, "train")
    save_captions(data_captions_train, data_path_out, version, "train")


    file_path = data_path + "/" +filename + "validation.h5"
    f = h5py.File(file_path, 'r')

    data_ims_test = create_features(f["input_image"], early_stop)
    data_captions_test = create_captions(f["input_name"], early_stop)
    f.close()

    save_images(data_ims_test, data_path_out, version, "test")
    save_captions(data_captions_test, data_path_out, version, "test")


def save_images(ims, data_path, version, split):
    if split == "train":
        filename = "{}/data_ims_{}_{}.npy".format(data_path, version, split)
    elif split == "test":
        filename = "{}/data_ims_{}_devtest.npy".format(data_path, version)

    np.save(filename, ims)
    print("Shape data ims {} is: {}".format(split, ims.shape))


def save_captions(captions, data_path, version, split):
    if split =="train":
        filename = '{}/data_captions_{}_{}.txt'.format(data_path, version, split)
    elif split == "test":
        filename = '{}/data_captions_{}_devtest.txt'.format(data_path, version, split)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for caption in captions:
            writer.writerow((caption[0], caption[1]))
    print("length {} captions is {}".format(split, len(captions)))

def create_captions(captions, early_stop):
    cleaned_captions = []
    count = 0
    for caption in captions:
        cleaned_captions.append((count, caption[0].decode("latin-1").lower()))
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

def segment_dresses_tile(img):
    segments = {}
    H, W, C = img.shape
    # 1=x_1, 2=y_1, 3=x_2, 4 =y_2 linkerbovenhoek=(x_1, y_1) rechteronderhoek=(x_2, y_2)
    bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                      [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)]])

    segments["top"] = img[: floor(0.33*H) , : floor(0.5*W) , :]
    segments["full_skirt"] = img[: floor(0.33*H) ,  floor(0.5*W): , :]
    segments["skirt_above_knee"] = img[floor(0.33*H): floor(0.66*H) , : floor(0.5*W) , :]
    segments["neckline"] = img[floor(0.33*H): floor(0.66*H) , floor(0.5*W):  , :]
    segments["left_sleeve"] = img[floor(0.66*H):  , : floor(0.5*W) , :]
    segments["right_sleeve"] = img[floor(0.66*H): , floor(0.5*W):  , :]
    segments["full"] = img

    return segments, bboxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="fashiongen_256_256_",
                        help='path to training file')
    parser.add_argument('--data_path', default="../../data/Fashion_gen",
                        help='path to data folder.')
    parser.add_argument('--data_path_out', default="../../data/Fashion_gen",
                    help='path to data folder.')
    parser.add_argument('--early_stop', default=None, type=int,
                        help='Rank loss margin.')
    parser.add_argument('--version', default=None,
                        help='version control')
    args = parser.parse_args()
    main(args)
