import evaluation
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import h5py

from util.segment_dresses import segment_dresses_tile
from vocab import Vocabulary, deserialize_vocab  # NOQA
from model import SCAN, xattn_score_t2i, xattn_score_i2t
from data_ken import get_test_loader
from evaluation import encode_data
import os
from utils import get_random_indx, str2bool, print_result_subset, calculate_r
import re
import sys
sys.path.append('../visualize')
from viz_utils import show_plots, get_matches_t2i, get_matches_i2t, check_caption_length, get_target_id, prepare_embeddings, get_indx_subset, get_id

"""
Visualize the attention of the SCAN model.
How does the model match words and image segments together
Also shows the best 5 matches for text and image queries within the test set
"""

def main(args):

    model_path = "{}/{}/seed1/checkpoint/{}".format(args.model_path, args.run, args.checkpoint )

    # load model and options
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    opt = checkpoint['opt']

    # add because basic is not present in model
    d = vars(opt)
    d['basic'] = False

    run = args.run
    data_path = "{}/{}".format(args.data_path, args.data_name)
    nr_examples = args.nr_examples
    version = opt.version
    clothing = opt.clothing

    if opt.trans:
        plot_folder = "plots_trans"
    else:
        plot_folder = "plots_scan"

    plot_path = '{}/{}_{}'.format(plot_folder, version, run)
    caption_test_path = "{}/{}/data_captions_{}_test.txt".format(data_path, clothing, version)
    image_path = "{}".format( args.image_folder)
    vocab_path = "{}/{}".format(args.vocab_path, clothing)
    data_folder = "../data"

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # change image paths from lisa folders to local folders
    opt.data_path = data_folder
    opt.image_path = image_path
    opt.vocab_path = vocab_path
    print(opt)

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    try:
        embs = torch.load("{}/embs/embs_{}_{}.pth.tar".format(plot_folder,run, version), map_location=('cpu'))
        print("loading embeddings")
        img_embs = embs["img_embs"]
        cap_embs = embs["cap_embs"]
        cap_lens = embs["cap_lens"]
        freqs = embs["freqs"]
    except:
        print("Create embeddings")
        img_embs, cap_embs, cap_lens, freqs = get_embs(opt, model ,run , version, data_path, plot_folder, vocab_path=vocab_path)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] , cap_embs.shape[0]))

    temp = torch.load("{}/ranks_{}_{}.pth.tar".format(plot_folder,run, version), map_location=('cpu'))

    rt = temp["rt"]
    rti = temp["rti"]
    attn = temp["attn"]
    t2i_switch = temp["t2i_switch"]

    r_i2t = calculate_r(rt[0], "i2t")
    r_t2i = calculate_r(rti[0], "t2i")

    top1_rt = rt[1]
    top1_rti = rti[1]

    if args.focus_subset:
        indx = get_indx_subset(caption_test_path, args.word_asked)
        rs_i2t = calculate_r(rt[0][indx], "i2t")
        rs_t2i = calculate_r(rti[0][indx], "t2i")
        print_result_subset(rs_i2t, r_i2t, "i2t", args.word_asked)
        print_result_subset(rs_t2i, r_t2i, "t2i", args.word_asked)
        rnd_indx = get_random_indx(nr_examples, len(indx))
        rnd = [indx[i] for i in rnd_indx]
    else:
        rnd = get_random_indx(nr_examples, len(top1_rt))

    # dictionary to turn test_ids to data_ids
    test_id2data = {}

    # find the caption and image with every id in the test file {caption_id : (image_id, caption)}
    with open(caption_test_path, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(caption_reader):
            test_id2data[i] = (line[0], line[1])

    h5_images = get_h5_images(args.data_name, data_path)

    # get the matches
    matches_i2t = get_matches_i2t(top1_rt, test_id2data, nr_examples, rnd)
    matches_t2i = get_matches_t2i(top1_rti, test_id2data, nr_examples, rnd)

    # get id for file name
    unique_id = get_id(plot_path, "i2t", run)

    # plot image and caption together
    show_plots(matches_i2t, len(matches_i2t), "i2t" , run, version, plot_path, args, clothing, h5_images, unique_id)
    show_plots(matches_t2i, len(matches_t2i), "t2i" , run, version, plot_path, args, clothing, h5_images, unique_id)

    for i in range(len(rnd)):
        wanted_id = rnd[i]
        target_id = get_target_id(top1_rt, top1_rti, t2i_switch, wanted_id)

        attn = get_attn(img_embs, cap_embs, cap_lens, wanted_id, target_id, opt, t2i_switch, freqs)

        if t2i_switch:
            words_caption = get_captions(test_id2data, wanted_id)
            image_segs = get_image_segs(target_id, test_id2data, args, opt, model, h5_images)
            match_t2i_viz(attn, wanted_id, target_id, test_id2data, run, version, plot_path, clothing, words_caption, image_segs)
        else:
            words_caption = get_captions(test_id2data, target_id)
            image_segs = get_image_segs(wanted_id, test_id2data, args, opt, model, h5_images)
            match_i2t_viz(attn, wanted_id, target_id, test_id2data, run, version, plot_path, words_caption, image_segs)


def get_h5_images(data_name, data_path):
    if data_name == "Fashion200K":
        return None
    elif data_name == "Fashion_gen":
        file = "{}/all/fashiongen_256_256_validation.h5".format(data_path)
        f = h5py.File(file, 'r')
        dset = f["input_image"]
        return dset

def get_image_segs(id, test_id2data, args, opt, model, h5_images):
    image_segs = []
    #t2i
    img_id_data = test_id2data[id][0]

    if args.data_name == "Fashion200K":
        img_adress = "{}/{}/{}/pictures_only/pictures_only/{}_0.jpeg".format(args.data_path,opt.data_name, opt.clothing,img_id_data)
        if opt.trans:
            image = Image.open(img_adress)
        else:
            image = mpimg.imread(img_adress)
    elif args.data_name == "Fashion_gen":
        image = h5_images[int(img_id_data)]
        if opt.trans:
            image = Image.fromarray(image)


    if opt.trans:
        if opt.rectangle:
            height = 512
        else:
            height = 256

        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.Resize((height, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)
        x = model.img_enc.stn(image)
        x = x.data.numpy()
        for i in range(x.shape[0]):
            temp = np.moveaxis(x[i], 0, -1)
            image_segs.append(temp)
    else:
        segs, _ = segment_dresses_tile(image)
        for key in segs.keys():
            image_segs.append(segs[key])

    return image_segs


def get_captions(test_id2data, id):
    caption = test_id2data[id][1]
    words_caption = re.split(r"[\s'%]", caption)
    words_caption.insert(0, "<START>")
    words_caption.append("<END>")
    return words_caption


def get_attn(img_embs, cap_embs, cap_lens, wanted_id, target_id, opt, t2i_switch, freqs):
    i, s,l = prepare_embeddings(img_embs, cap_embs, cap_lens, wanted_id, target_id, t2i_switch)

    sim, attn = xattn_score_t2i(i, s, l, freqs, opt)
    return attn


def get_embs(opt, model ,run , version, data_path, plot_folder, split='test', fold5=False, vocab_path="../vocab/"):
    # load vocabulary used by the model
    vocab = deserialize_vocab("{}/{}_vocab_{}.json".format(vocab_path, opt.data_name, version))
    opt.vocab_size = len(vocab)

    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    img_embs, cap_embs, cap_lens, freqs = encode_data(model, data_loader)

    if not os.path.exists('{}/embs'.format(plot_folder)):
        os.makedirs('{}/embs'.format(plot_folder))

    torch.save({'img_embs': img_embs, 'cap_embs': cap_embs, "cap_lens": cap_lens, "freqs": freqs}, '{}/embs/embs_{}_{}.pth.tar'.format(plot_folder, run, version))
    print("Saved embeddings")
    return img_embs, cap_embs, cap_lens, freqs

def match_i2t_viz(attn, wanted_id, target_id, test_id2data, run, version, plot_path, words_caption, image_segs):
    # store the probability vector for every image region in the image regarding the caption
    img_prob = {}

    image_len = len(image_segs)
    # get the caption length
    cap_len = attn[0].shape[2]

    if not check_caption_length(cap_len, words_caption):
        print("Captions lengths are not equal")
        exit()

    img_id_data = test_id2data[wanted_id][0]

    # find the probability of every image region for each word
    for i in range(image_len):
        img_prob[i] = attn[0][0, i, :]

    fig, axs = plt.subplots(image_len, cap_len + 1, figsize=(20,20))

    for i in range(image_len):
        for j in range(cap_len + 1):
            highest_indx = np.argmax(img_prob[i])
            lowest_indx = np.argmin(img_prob[i])
            if j == 0:
                axs[i, j].imshow(image_segs[i])
                axs[i, j].axis("off")
            else:
                axs[i, j].text(0.5, 0.5, words_caption[j - 1], size=20)
                if j == highest_indx + 1:
                    axs[i, j].set_title( '%.3f'%(img_prob[i][j - 1].item()), fontweight="bold", color="green")
                elif j == lowest_indx + 1:
                    axs[i, j].set_title( '%.3f'%(img_prob[i][j - 1].item()), fontweight="bold", color="red")
                else:
                    axs[i, j].set_title( '%.3f'%(img_prob[i][j - 1].item()))
                axs[i, j].axis("off")

    plt.show()
    plt.savefig('{}/viz_attention{}_{}_{}.png'.format(plot_path, wanted_id, run, version))
    plt.close(fig)


def match_t2i_viz(attn, wanted_id, target_id, test_id2data, run, version, plot_path, clothing, words_caption, image_segs):
    word_prob = {}

    # check the caption length
    image_len = len(image_segs)
    cap_len = attn[0].shape[2]
    if not check_caption_length(cap_len, words_caption):
        print("Captions lengths are not equal")
        exit()

    # find the probability of every image region for each word
    for i in range(cap_len):
        word_prob[i] = attn[0][0, :, i]

    fig, axs = plt.subplots(cap_len, image_len + 1, figsize=(20,20))

    for i in range(cap_len):
        for j in range(image_len + 1):
            highest_indx = np.argmax(word_prob[i])
            lowest_indx = np.argmin(word_prob[i])

            if j == 0:
                axs[i, j].text(0.5, 0.5, words_caption[i], size=20)
                axs[i, j].axis("off")
            else:
                axs[i, j].imshow(image_segs[j-1])
                if j == highest_indx + 1:
                    axs[i, j].set_title( '%.3f'%(word_prob[i][j - 1].item()), fontweight="bold", color="green")
                elif j == lowest_indx + 1:
                    axs[i, j].set_title( '%.3f'%(word_prob[i][j - 1].item()), fontweight="bold", color="red")
                else:
                    axs[i, j].set_title( '%.3f'%(word_prob[i][j - 1].item()))
                axs[i, j].axis("off")

    # plt.show()
    plt.savefig('{}/viz_attention{}_{}_{}.png'.format(plot_path, wanted_id, run, version))
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained SCAN model')
    parser.add_argument('--run', default="Run2", type=str, help='Which run to evaluate')
    parser.add_argument('--checkpoint', default="model_best.pth.tar", type=str, help='which checkpoint to use')
    parser.add_argument('--model_path', default="runs", type=str, help='which checkpoint to use')
    parser.add_argument('--nr_examples', default=4, type=int, help="nr of examples to be plot")
    parser.add_argument('--data_path', default="../data", type=str, help='which checkpoint to use')
    parser.add_argument('--data_name', default="Fashion200K", type=str, help='which data set to use')
    parser.add_argument('--vocab_path', default="../vocab", type=str, help='which checkpoint to use')
    parser.add_argument('--word_asked', default="", type=str, help='focus on certain word based on subset')
    parser.add_argument('--focus_subset', default=False, type=str2bool, help='create a subset of data using a word')
    parser.add_argument('--data_folder', default="../data", type=str, help='to datafolder')
    parser.add_argument('--image_folder', default="pictures_only/pictures_only", type=str, help='path to images')

    args = parser.parse_args()
    main(args)
