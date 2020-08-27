
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from torch.autograd import Variable
import torch
import re

import numpy as np
import glob



# show the matches of images and captions
def show_plots(matches, n_example, title, run, version, plot_path, args, clothing, h5_images, unique_id):
    w = 10
    h = 10
    fig = plt.figure(figsize=(40, 20))
    fig.tight_layout()

    columns = n_example
    rows = 1

    # prep (x,y) for extra plotting
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # ax enables access to manipulate each of subplots
    ax = []

    j=0

    for i in range(len(matches)):
        if args.data_name == "Fashion200K":
            temp = "{}/Fashion200K/women/{}/**/{}/*_0.jpeg".format(args.data_folder, clothing ,matches[i][0])
            img_adress = glob.glob(temp)
            img = mpimg.imread(img_adress[0])
        elif args.data_name == "Fashion_gen":
            img = h5_images[int(matches[i][0])]

        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        j+=1
        ax[-1].set_title(title + ":" + matches[i][1] + " ({})".format(matches[i][2]),fontsize=22 )  # set title
        plt.imshow(img)

    plt.savefig('{}/{}_save_plots_{}_{}.png'.format(plot_path, unique_id, title, run))
    plt.close(fig)


def get_matches_t2i(top1, test_id2data, nr_examples, rnd):
    matches = []

    for i in range(len(rnd)):
        image_id = top1[rnd[i]]

        match = "Incorrect"
        if rnd[i] == image_id:
            match = "correct"
        image_id = test_id2data[image_id][0]
        caption = test_id2data[rnd[i]][1]

        matches.append((image_id, caption, match))
    return matches


def get_matches_i2t(top1, test_id2data, nr_examples, rnd):
    matches = []

    # for every image find caption > i2t
    for i in range(len(rnd)):
        caption_id = top1[rnd[i]]
        caption = test_id2data[caption_id][1]
        image_id = test_id2data[rnd[i]][0]
        match = "Incorrect"
        if rnd[i] == caption_id:
            match = "correct"
        matches.append((image_id, caption, match))
    return matches


def check_caption_length(caplen, caption):
    if len(caption)  != caplen:
        return False
    else:
        return True

def get_target_id(rt, rti, t2i_switch, wanted_id):
    if t2i_switch:
        target_id = int(rti[wanted_id])
    else:
        target_id = int(rt[wanted_id])
    return target_id

def prepare_embeddings(img_embs, cap_embs, cap_lens, wanted_id, target_id, t2i_switch):
    if t2i_switch:
        i = np.expand_dims(img_embs[target_id], axis=0)
        s = np.expand_dims(cap_embs[wanted_id], axis=0)
        l = [cap_lens[wanted_id]]
    else:
        i = np.expand_dims(img_embs[wanted_id], axis=0)
        s = np.expand_dims(cap_embs[target_id], axis=0)
        l = [cap_lens[target_id]]

    i = Variable(torch.from_numpy(i))
    s = Variable(torch.from_numpy(s))
    return i, s, l


def get_indx_subset(caption_test_path, word_asked):

    indx = []
    with open(caption_test_path, newline = '') as games:
        reader = csv.reader(games, delimiter='\t')
        for i , line in enumerate(reader):
            if word_asked in line[1].split():
                indx.append(i)
    return indx

def get_id(plot_path, title, run):
    runs = glob.glob('{}/*_save_plots_{}_{}.png'.format(plot_path,title, run))
    nr_runs = len(runs)
    return nr_runs + 1
