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
sys.path.append('../../')
sys.path.append('/home/kgoei/thesis/comb/util')

from segment_dresses import segment_dresses, segment_dresses_tile, segment_dresses_tile_nine
# from simCLR.models.resnet_simclr import ResNetSimCLR
from Layers_simCLR_pre import Layers_simCLR_pre
from layers_model import LayersModel
from Layers_resnest import Layers_resnest
from layers_alex2 import LayersModelAlex
from DeepFashion import LayersAttr


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
def get_captions(args):
    data_captions = {}
    image_ids_captions =[]
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
                data_captions[img_id] = description
                image_ids_captions.append((caption[0], img_id))
        file.close()

    print("get_captions: ", len(data_captions))
    return data_captions, image_ids_captions

# segment in seven parts and push through net
def get_features(img, net, img_idx, transform, segments, bboxes, device, network ):
    W, H, C = img.shape

    if network == "layers" or network == "layers2":
        img = Image.fromarray(img)
        img_transformed = transform(img).unsqueeze(0).to(device)
        features = net.forward1(img_transformed).to("cpu").squeeze()
    elif network == "layers_resnest" or network == "layers_simCLR" or "layers_attr":
        img = Image.fromarray(img)
        img_transformed = transform(img).unsqueeze(0).to(device)
        features = net.forward(img_transformed).to("cpu").squeeze()

    elif network == "vilbert":
        stacked_segs = stack_segments(segments, transform)
        stacked_segs = stacked_segs.to(device)

        dict = net(stacked_segs)
        hidden_features = dict["out"]
        dim = hidden_features.shape[2]
        pool = nn.AvgPool2d((dim, dim))
        features = pool(hidden_features).to("cpu").squeeze()
    else:
        # stack segments to push through net
        stacked_segs = stack_segments(segments, transform)
        stacked_segs = stacked_segs.to(device)
        features = net(stacked_segs).to("cpu")
        if network == "alex":
            features = features.squeeze()
        # elif args.network == "simCLR" or args.network == "simCLR_pre":
        #     features = features[0].squeeze()

    features = features.detach().numpy()
    return {
        "image_id": int(img_idx),
        "image_h": int(H),
        "image_w": int(W),
        "num_boxes": int(features.shape[0]),
        "boxes" :base64.b64encode(bboxes),
        "features": base64.b64encode(features)
    }

def stack_segments(segments, transform):
    segs = []
    for key in segments:
        seg_pil = Image.fromarray(segments[key])

        # transform images
        seg_transformed = transform(seg_pil)

        segs.append(seg_transformed)

    stacked_segments = torch.stack(segs, dim=0)
    return stacked_segments

def get_model(args, device):
    if args.network == "layers":
        net = LayersModel(trained_dresses=args.trained_dresses, checkpoint_path=args.checkpoint)
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "layers2":
        net = LayersModel2(trained_dresses=args.trained_dresses, checkpoint_path=args.checkpoint)
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    elif args.network == "alex":
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
    elif args.network == "layers_resnest":
        # choose model
        net = Layers_resnest(feature_dim=2048, trained_dresses=args.trained_dresses, checkpoint_path=args.checkpoint)



        # set to evaluation
        net.eval()

        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "layers_attr":
        # choose model
        net = LayersAttr()
        # set to evaluation
        net.eval()

        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "sixth":
        # choose model
        net = models.alexnet(pretrained=True)
        # take aways the last layers
        net.classifier = nn.Sequential(*[net.classifier[i] for i in range(2)])

        # set to evaluation
        net.eval()

        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif args.network == "vilbert":
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
    elif args.network == "layers_simCLR":
        net = Layers_simCLR_pre(args, device, feature_dim=2048)
        net.eval()

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.ToTensor()])


    # elif args.network == "simCLR_pre":
    #     net = SimCLR(args)
    #     net.load_state_dict(torch.load(args.checkpoint_simCLR_pre, map_location=torch.device(device)))
    #     net.eval()
    #
    #     transform = transforms.Compose([
    #         transforms.RandomResizedCrop(size=(224, 224)),
    #         transforms.ToTensor()])
    #
    net = net.to(device)

    return net, transform

def generate_data(image_ids, args, net, transform, device):

    data = {}
    count_stop = 0
    # open file to write data to TSV

    print("Started reading images")
    # for every image create seven segmentations and push them through pretrained net
    for img_id in image_ids:
        path = img_id[0]
        img_idx = img_id[1]

        img_path = "{}/{}".format(args.data_dir, path)
        img = mpimg.imread(img_path)

        # segment dresses and retreive segmentations
        if args.tile:
            segments, bboxes = segment_dresses_tile(img)
        else:
            segments, bboxes = segment_dresses(img)

        # create features from segmentations
        seg = get_features(img, net, img_idx, transform, segments, bboxes, device, args.network)

        temp = np.frombuffer( base64.b64decode(seg["features"]), dtype=np.float32)
        temp = temp.reshape((seg["num_boxes"],-1))

        data[img_idx] = temp
        count_stop +=1
        if count_stop == args.early_stop:
            break

        if count_stop % 10 == 0:
            print(count_stop)
    return data

# create numpy array of features and txt file for captions, taking sure they match
def combine_data_captions(data, data_captions, image_ids, args):

    ids_needed = []
    for id in image_ids:
        adress, im_id = id
        if im_id in data.keys():
            ids_needed.append(im_id)
    data_out = np.stack([data[id] for id in ids_needed], axis=0)

    # check if dir already excists
    if not os.path.exists(args.data_out):
        os.makedirs(args.data_out)

    # print some shape checks
    print("Shape of data_out is {}".format(data_out.shape))
    print(len(ids_needed))
    if data_out.shape[0] != len(ids_needed):
        print("length should be equal!!")
        exit()

    # save images and captions
    np.save( "{}/data_ims_{}.npy".format(args.data_out, args.version), data_out)

    with open('{}/data_captions_{}.txt'.format(args.data_out, args.version), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for id in ids_needed:
            writer.writerow((id, data_captions[id]))

    return


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--early_stop',help='take lower number of samples for testing purpose', default=None, type=int)
    parser.add_argument('--version',help='add version', default=None, type=str)
    parser.add_argument('--network',help='alex|simCLR|simCLR_pre|layers|sixth|vilbert', default="alex", type=str)
    parser.add_argument('--data_dir',help='location data directory', default="../../data/Fashion200K", type=str)
    parser.add_argument('--data_out',help='location of data out', default="../../data/Fashion200K/dresses", type=str)
    parser.add_argument('--tile', action='store_true', help="use basic tile segmentation")
    parser.add_argument('--multi', action='store_true', help="use to create features for multi-modal evaluation")
    parser.add_argument('--clothing',help='clothing item', default="dresses", type=str)
    parser.add_argument("--list_clothing", nargs="+", default=["dresses"])

    # TO load pretrained dresses model
    parser.add_argument('--trained_dresses', action='store_true', help="load models trained on dresses")
    parser.add_argument('--checkpoint',help='path to saved model', default="../../train_models/runRes/train1/checkpoint/model_best.pth.tar", type=str)

    # WHEN PRETRAINED simCLR IS USED
    parser.add_argument('--output_dim',help='if simCLR is used, size of output dim', default=256, type=int)
    parser.add_argument('--input_shape_width', default=96, type=int, help='(W, H, C) for when pretrained network is used' )
    parser.add_argument('--input_shape_height', default=192, type=int, help='(W, H, C) for when pretrained network is used')
    parser.add_argument('--name_run',help='folder name of run', default=None, type=str)

    # WHEN simCLR_pre IS USED
    parser.add_argument('--resnet',help='which resnet to use', default="resnet50", type=str)
    parser.add_argument('--normalize',help='use normalize', default="True", type=str2bool)
    parser.add_argument('--projection_dim',help='size of projection dim', default=64, type=int)
    parser.add_argument('--checkpoint_simCLR_pre',help='location pretrained model', default="../../SimCLR_pre/checkpoint_100.tar", type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Called with args:')
    print(args)


    # get the ids and the captions from the text file for normal cross-modal training
    data_captions, image_ids = get_captions(args)

    # retrieve requiered model with correct transfrom
    net, transform = get_model(args, device)

    # generate features
    data = generate_data(image_ids, args, net, transform, device)

    # create numpy stack from features and match with captions
    combine_data_captions(data, data_captions, image_ids, args)
