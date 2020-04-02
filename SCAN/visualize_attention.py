from vocab import Vocabulary
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

def main(args):
    caption_test_path = args.caption_test_path
    run = args.run
    checkpoint = args.checkpoint
    data_path = args.data_path
    nr_examples = args.nr_examples
    version = args.version

    caption_test_path = "{}/data_captions_{}_test.txt".format(caption_test_path, version)
    model_path = "{}{}/checkpoint/{}".format(args.model_path, run,checkpoint )
    plot_path = "{}{}/checkpoint/".format(args.plot_path,  run)
    temp = torch.load("plots_scan/ranks_{}_{}.pth.tar".format(run, version))

    rt = temp["rt"]
    rti = temp["rti"]
    attn = temp["attn"]
    t2i_switch = temp["t2i_switch"]

    # dictionary to turn test_ids to data_ids
    test_id2data = {}

    # find the caption and image with every id in the test file {caption_id : (image_id, caption)}
    with open(caption_test_path, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(caption_reader):
            test_id2data[i] = (line[0], line[1])

    # get the matches
    matches_i2t = get_matches_i2t(rt[1], test_id2data, nr_examples)
    matches_t2i = get_matches_t2i(rti[1], test_id2data, nr_examples)

    # plot image and caption together
    show_plots(matches_i2t, len(matches_i2t), "i2t", plot_path, run)
    show_plots(matches_t2i, len(matches_t2i), "t2i", plot_path, run)

    if t2i_switch:
        match_t2i_viz(attn, 2,rti[1], test_id2data, run, version)
    else:
        match_i2t_viz(attn, 0, rt[1], test_id2data, run, version)

def match_i2t_viz(attn, wanted_image, matches_i2t, test_id2data, run, version):
    # store the probability vector for every image region in the image regarding the caption
    img_prob = {}

    # highest matched caption for image
    matched_cap_id = int(matches_i2t[wanted_image])

    # get the caption from the matched caption id
    caption = test_id2data[matched_cap_id][1]
    words_caption = caption.split(" ")
    words_caption.insert(0, "<START>")
    words_caption.append("<END>")

    # get the caption length
    cap_len = attn[matched_cap_id].shape[1]


    if not check_caption_length(cap_len, caption):
        print("Captions lengths are not equal")
        exit()

    img_id_data = test_id2data[wanted_image][0]

    for i in range(len(attn)):
        print(attn[i].shape)
    # find the probability of every image region for each word
    for i in range(7):
        img_prob[i] = attn[matched_cap_id][wanted_image, :, i]

    fig, axs = plt.subplots(7, cap_len + 1, figsize=(20,20))

    for i in range(7):
        for j in range(cap_len + 1):
            if j == 0:
                img_adress = glob.glob("{}Fashion200K/dresses_segmentations/{}/*_{}.jpeg".format(args.data_path ,img_id_data, i))
                img = mpimg.imread(img_adress[0])
                axs[i, j].imshow(img)
                axs[i, j].axis("off")
            else:
                axs[i, j].text(0.5, 0.5, words_caption[j - 1], size=20)
                axs[i, j].set_title( '%.3f'%(img_prob[i][j - 1].item()))
                axs[i, j].axis("off")

    plt.show()
    plt.savefig('plots_scan/viz_attention_{}_{}.png'.format(run, version))
    plt.close(fig)



def match_t2i_viz(attn, wanted_caption, matches_t2i, test_id2data, run, version):
    # store the probability vector for every word in the caption regarding the image regions
    word_prob = {}

    # highest matched image for caption
    matched_img_id = int(matches_t2i[wanted_caption])

    # get the caption from the wanted caption id
    caption = test_id2data[wanted_caption][1]
    words_caption = caption.split(" ")
    words_caption.insert(0, "<START>")
    words_caption.append("<END>")
    # get the caption length
    cap_len = attn[wanted_caption].shape[2]

    if not check_caption_length(cap_len, caption):
        print("Captions lengths are not equal")
        exit()

    img_id_data = test_id2data[matched_img_id][0]


    # find the probability of every image region for each word
    for i in range(cap_len):
        word_prob[i] = attn[wanted_caption][matched_img_id, :, i]

    fig, axs = plt.subplots(cap_len, 8, figsize=(20,20))

    for i in range(cap_len):
        for j in range(8):
            if j == 0:
                axs[i, j].text(0.5, 0.5, words_caption[i], size=20)
                axs[i, j].axis("off")
            else:
                img_adress = glob.glob("{}Fashion200K/dresses_segmentations/{}/*_{}.jpeg".format(args.data_path ,img_id_data, j - 1))
                img = mpimg.imread(img_adress[0])
                axs[i, j].imshow(img)
                axs[i, j].set_title( '%.3f'%(word_prob[i][j - 1].item()))
                axs[i, j].axis("off")

    plt.show()
    plt.savefig('plots_scan/viz_attention_{}_{}.png'.format(run, version))
    plt.close(fig)


def check_caption_length(caplen, caption):
    length_sentence = len(caption.split(" "))
    if length_sentence + 2 != caplen:
        return False
    else:
        return True

def get_matches_i2t(top1, test_id2data, nr_examples):
    matches = []

    if nr_examples > len(top1):
        nr_examples = len(top1)

    # for every image find caption > i2t
    for i in range(nr_examples):
        caption_id = top1[i]
        caption = test_id2data[caption_id][1]
        image_id = test_id2data[i][0]
        match = "Incorrect"
        if i == caption_id:
            match = "correct"
        matches.append((image_id, caption, match))
    return matches


def get_matches_t2i(top1, test_id2data, nr_examples):
    matches = []

    if nr_examples > len(top1):
        nr_examples = len(top1)
    # for every image find caption > i2t
    for i in range(nr_examples):
        image_id = top1[i]

        match = "Incorrect"
        if i == image_id:
            match = "correct"
        image_id = test_id2data[image_id][0]
        caption = test_id2data[i][1]

        matches.append((image_id, caption, match))
    return matches


# show the matches of images and captions
def show_plots(matches, n_example, title, plot_path, run):
    w = 10
    h = 10
    fig = plt.figure(figsize=(20, 20))
    columns = n_example
    rows = 1

    # prep (x,y) for extra plotting
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # ax enables access to manipulate each of subplots
    ax = []

    j=0

    for i in range(len(matches)):
        img_adress = glob.glob("{}Fashion200K/women/dresses/**/{}/*_0.jpeg".format(args.data_path ,matches[i][0]))
        img = mpimg.imread(img_adress[0])

        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        j+=1
        ax[-1].set_title(title + ":" + matches[i][1] + " ({})".format(matches[i][2]))  # set title
        plt.imshow(img)


    plt.savefig('plots_scan/save_plots_{}_{}.png'.format(title, run))
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained SCAN model')

    parser.add_argument('--caption_test_path', default="../data/Fashion200K", type=str, help='path to captions')
    parser.add_argument('--run', default="Run0", type=str, help='Which run to evaluate')
    parser.add_argument('--checkpoint', default="model_best.pth.tar", type=str, help='which checkpoint to use')
    parser.add_argument('--model_path', default="runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--nr_examples', default=5, type=int, help="nr of examples to be plot")
    parser.add_argument('--data_path', default="../data/", type=str, help='which checkpoint to use')
    parser.add_argument('--vocab_path', default="../vocab/", type=str, help='which checkpoint to use')
    parser.add_argument('--plot_path', default="/$HOME/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--version', default="laenen", type=str, help='which version of features and vocab to use')

    args = parser.parse_args()
    main(args)
