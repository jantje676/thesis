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
import sys
sys.path.append('../../')
sys.path.append('/home/kgoei/thesis/comb/util')
from torchvision import transforms
import nltk
from DeepFashion import LayersAttr
from Layers_simCLR_pre import Layers_simCLR_pre
from layers_model import LayersModel
from Layers_resnest import Layers_resnest
from segment_dresses import segment_dresses


def main(args):
    filename = args.filename
    early_stop = args.early_stop
    data_path = args.data_path
    version = args.version
    data_path_out = args.data_path_out
    only_text = args.only_text
    description = args.descriptions
    network = args.network
    checkpoint = args.checkpoint
    trained_dresses = args.trained_dresses

    file_path = data_path + "/" +filename + "train.h5"
    f = h5py.File(file_path, 'r')

    if args.descriptions:
        field = "input_description"
    else:
        field = "input_name"

    if not only_text:
        data_ims_train = create_features(f["input_image"], early_stop, network, trained_dresses, checkpoint)
        save_images(data_ims_train, data_path_out, version, "train")

    data_captions_train = create_captions(f[field], early_stop, description)
    save_captions(data_captions_train, data_path_out, version, "train")

    f.close()

    file_path = data_path + "/" +filename + "validation.h5"

    f = h5py.File(file_path, 'r')
    if not only_text:
        data_ims_test = create_features(f["input_image"], early_stop, network, trained_dresses, checkpoint)
        save_images(data_ims_test, data_path_out, version, "test")

    data_captions_test = create_captions(f[field], early_stop, description)
    save_captions(data_captions_test, data_path_out, version, "test")
    f.close()

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

def create_captions(captions, early_stop, description):
    cleaned_captions = []
    count = 0
    for i in range(len(captions)):
        caption = captions[i][0].decode("latin-1").lower()

        if description:
            seg_cap = caption.split(".")
            words = nltk.word_tokenize(seg_cap[0])
            words = [word.lower() for word in words if word.isalpha()]
            if len(words) == 0:
                words = ["fashion"]
            caption = " ".join(words)
        cleaned_captions.append((count, caption))

        count += 1
        if early_stop == count and early_stop != None:
            break
    return cleaned_captions

def create_features(images, early_stop, network, trained_dresses, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net, transform = get_model(network, trained_dresses, checkpoint)

    net = net.to(device)

    features = []
    count = 0
    for image in images:
        if network == "layers" or network == "layers_resnest":
            img = Image.fromarray(image)
            img_transformed = transform(img).unsqueeze(0).to(device)
            feature = net.forward1(img_transformed).to("cpu").squeeze().detach().numpy()
            features.append(feature)
        elif network == "layers_simCLR" or "layers_attr":
            img = Image.fromarray(image)
            img_transformed = transform(img).unsqueeze(0).to(device)
            feature = net.forward(img_transformed).to("cpu").squeeze().detach().numpy()
            features.append(feature)
        elif network == "vilbert":
            segments, bboxes = segment_dresses(image)
            stacked_segs = stack_segments(segments, transform)
            stacked_segs = stacked_segs.to(device)
            dict = net(stacked_segs)
            hidden_features = dict["out"]
            dim = hidden_features.shape[2]
            pool = nn.AvgPool2d((dim, dim))
            feature = pool(hidden_features).to("cpu").squeeze().detach().numpy()
            features.append(feature)
        else:
            # create segments in a dictionary
            segments, bboxes = segment_dresses(image)
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



def get_model(network, trained_dresses, checkpoint):
    if network == "alex":
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
    elif network == "layers":
        net = LayersModel(trained_dresses=trained_dresses, checkpoint_path=checkpoint)
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "layers_attr":
        # choose model
        net = LayersAttr(args.checkpoint)
        # set to evaluation
        net.eval()

        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "layers_simCLR":
        net = Layers_simCLR_pre(args, device, feature_dim=2048)
        net.eval()

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor()])
    elif network == "vilbert":
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
    elif network == "layers_resnest":
        # choose model
        net = Layers_resnest(img_dim=2048, trained_dresses=trained_dresses, checkpoint_path=checkpoint)
        # set to evaluation

        net.eval()

        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
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
    parser.add_argument('--filename', default="fashiongen_256_256_",
                        help='path to training file')
    parser.add_argument('--data_path', default="../../data/Fashion_gen/all",
                        help='path to data folder.')
    parser.add_argument('--data_path_out', default="../../data/Fashion_gen/all",
                    help='path to data folder.')
    parser.add_argument('--early_stop', default=None, type=int,
                        help='Rank loss margin.')
    parser.add_argument('--version', default=None,
                        help='version control')
    parser.add_argument('--only_text', action='store_true',
                        help="only create new captions")
    parser.add_argument('--descriptions', action='store_true',
                        help="create captions from the input_descriptions field of the data")
    parser.add_argument('--network',help='alex|layers|layers_resnest', default="alex", type=str)

    # TO load pretrained dresses model
    parser.add_argument('--trained_dresses', action='store_true', help="load models trained on dresses")
    parser.add_argument('--checkpoint',help='path to saved model', default="../../train_models/runRes/train1/checkpoint/model_best.pth.tar", type=str)


    args = parser.parse_args()
    main(args)
