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

"Probabaly there is still a bug in the t2i matching process. \
The t2i recall shows that  there is 25% match rate, but pictures\
but show that there is no match rate"

def main(args):
    print(args.version)
    caption_test_path = args.caption_test_path
    run = args.run
    checkpoint = args.checkpoint
    data_path = args.data_path
    nr_examples = args.nr_examples
    version = args.version

    caption_test_path = "{}/data_captions_{}_test.txt".format(caption_test_path, version)
    model_path = "{}{}/checkpoint/{}".format(args.model_path, run,checkpoint )
    plot_path = "{}{}/checkpoint/".format(args.plot_path,  run)
    rt, rti = evaluation.evalrank(model_path, plot_path, run, version, data_path=args.data_path, split="test", vocab_path=args.vocab_path)

    # dictionary to turn test_ids to data_ids
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
    show_plots(matches_i2t, len(matches_i2t), "i2t", plot_path, run)
    show_plots(matches_t2i, len(matches_t2i), "t2i", plot_path, run)



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
        image_id = test_id2data[image_id][0]
        caption = test_id2data[i][1]
        match = "Incorrect"
        if i == image_id:
            match = "correct"
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

    parser.add_argument('--caption_test_path', default="/$TMPDIR/thesis/data/Fashion200K", type=str, help='path to captions')
    parser.add_argument('--run', default="Run0", type=str, help='Which run to evaluate')
    parser.add_argument('--checkpoint', default="model_best.pth.tar", type=str, help='which checkpoint to use')
    parser.add_argument('--model_path', default="/$TMPDIR/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--nr_examples', default=5, type=int, help="nr of examples to be plot")
    parser.add_argument('--data_path', default="/$TMPDIR/thesis/data/", type=str, help='which checkpoint to use')
    parser.add_argument('--vocab_path', default="/$TMPDIR/thesis/vocab/", type=str, help='which checkpoint to use')
    parser.add_argument('--plot_path', default="/$HOME/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--version', default="laenen", type=str, help='which version of features and vocab to use')

    args = parser.parse_args()
    main(args)
