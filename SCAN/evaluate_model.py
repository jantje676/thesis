from vocab import Vocabulary
import evaluation
import csv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

def main():

    caption_test_path = "../data/Fashion200K/data_captions_test.txt"
    run = "runX"
    checkpoint = "model_best.pth.tar"
    nr_examples = 2

    rt, rti = evaluation.evalrank("runs/{}/checkpoint/{}".format(run, checkpoint), data_path="../data/", split="test")
    # rt, rti =  evaluation.evalrank("runs/runX/checkpoint/checkpoint_3.pth.tar", data_path="../data/", split="test")

    # rt = (ranks, top1)
    # tuple (image_id, caption)
    test_id2data = {}

    # find the caption and image with every id in the test file
    with open(caption_test_path, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(caption_reader):
            test_id2data[i] = (line[0], line[1])

    # get the matches
    matches_i2t = get_matches_i2t(rt[1], test_id2data, nr_examples)
    matches_t2i = get_matches_t2i(rti[1], test_id2data, nr_examples)

    # plot image and caption together
    show_plots(matches_i2t, len(matches_i2t), "i2t")
    show_plots(matches_t2i, len(matches_t2i), "t2i")



def get_matches_i2t(top1, test_id2data, nr_examples):
    matches = []

    if nr_examples > len(top1):
        nr_examples = len(top1)

    # for every image find caption > i2t
    for i in range(nr_examples):
        caption_id = top1[i]
        caption = test_id2data[caption_id][1]
        image_id = test_id2data[i][0]
        matches.append((image_id, caption))
    return matches


def get_matches_t2i(top1, test_id2data, nr_examples):
    matches = []

    if nr_examples > len(top1):
        nr_examples = len(top1)
    # for every image find caption > i2t
    for i in range(nr_examples):
        image_id = top1[i]
        image_id = test_id2data[image_id][0]
        caption = test_id2data[i][1]
        matches.append((image_id, caption))
    return matches


# show the matches of images and captions
def show_plots(matches, n_example, title):
    w = 10
    h = 10
    fig = plt.figure(figsize=(10, 5))
    columns = n_example
    rows = 1

    # prep (x,y) for extra plotting
    xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))           # absolute of sine

    # ax enables access to manipulate each of subplots
    ax = []

    j=0

    for i in range(len(matches)):
        img_adress = glob.glob("../data/Fashion200K/women/dresses/**/{}/*_0.jpeg".format(matches[i][0]))
        img = mpimg.imread(img_adress[0])

        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        j+=1
        ax[-1].set_title(title + ":" + matches[i][1])  # set title
        plt.imshow(img)

    plt.show()  # finally, render the plot



if __name__ == "__main__":
    main()
