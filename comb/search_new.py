from vocab import Vocabulary
from evaluation import *
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import argparse
import torch
"""
Evaluate the trained model on the test set
"""


# python search_new.py --model_path "runs/" --run run28 --vocab_path "../vocab/" --data_path "../data/"

def main(args):
    model_path = "{}{}/seed1/checkpoint/{}".format(args.model_path, args.run, args.checkpoint )
    find_sims(model_path, args.run, args.top_n, data_path=args.data_path, split="test", vocab_path=args.vocab_path, change=args.change)

def find_sims(model_path,run, n, data_path=None, split='dev', fold5=False, vocab_path="../vocab/", change=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)

    # add because div_transform is not present in model
    # d = vars(opt)
    # d['tanh'] = True

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab("{}{}/{}_vocab_{}.json".format(vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)
    print(opt.vocab_size)
    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    if change:
        opt.clothing = "dresses"

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens, freqs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] , cap_embs.shape[0]))

    t2i_switch = True
    if opt.cross_attn == 't2i':
        sims, attn = shard_xattn_t2i(img_embs, cap_embs, cap_lens, freqs, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims, attn = shard_xattn_i2t(img_embs, cap_embs, cap_lens, freqs, opt, shard_size=128)
        t2i_switch = False
    else:
        raise NotImplementedError

    top_t = i2t(img_embs, cap_embs, cap_lens, sims, n, return_ranks=True)
    top_i = t2i(img_embs, cap_embs, cap_lens, sims, n, return_ranks=True)
    img_idx, captions = get_data(opt)

    plot_text(top_t, img_idx, captions,opt, run)
    plot_image(top_i, img_idx, captions, opt, run, n)


def plot_image(top_i, img_idx, captions,opt, run, n):
    indices = [4,17,26]
    dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)

    query_caps = []
    img_adresses = []
    for i in range(len(indices)):
        query_caps.append(captions[indices[i]])
        images = top_i[indices[i]]
        for id in images:
            img_adresses.append("{}/pictures_only/pictures_only/{}_0.jpeg".format(dpath, img_idx[id]))

    fig = plt.figure(figsize=(40, 20))
    fig.tight_layout()

    columns = n
    rows = int(len(img_adresses)/n)
    ax = []


    for i in range(columns*rows):

        img = mpimg.imread(img_adresses[i])

        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1))
        plt.imshow(img)
        plt.axis('off')


    output_folder = "plots_scan/{}".format(run)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig('{}/image_retrieval_{}.png'.format(output_folder, str(indices[0])))
    plt.close(fig)


    file = open("{}/image_retrieval_{}.txt".format(output_folder, str(indices[0]), "w"))

    for i, caps in enumerate(query_caps):
        indx = indices[i]
        file.write("IMAGE {}: {} \n".format(indx, caps))
        for j in range(len(top_i[indx])):
            if indx == top_i[indx][j]:
                file.write("CORRECT IMAGE IS: {}".format(j))
        file.write("\n \n \n \n ")
    file.close()



def plot_text(top_t, img_idx, captions,opt, run):
    indices = [4,17,26]

    dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)

    img_adresses = []
    cap_for_image = []
    for i in range(len(indices)):
        img_adresses.append("{}/pictures_only/pictures_only/{}_0.jpeg".format(dpath, img_idx[indices[i]]))
        cap_idx = top_t[indices[i]]
        temp = []
        for id in cap_idx:
            temp.append(captions[id])
        cap_for_image.append(temp)


    fig = plt.figure(figsize=(40, 20))
    fig.tight_layout()

    columns = len(img_adresses)
    rows = 1
    ax = []

    for j,img_adress in enumerate(img_adresses):
        img = mpimg.imread(img_adress)

        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, j+1))
        plt.imshow(img)
        plt.axis('off')

    output_folder = "plots_scan/{}".format(run)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig('{}/text_retrieval_{}.png'.format(output_folder,str(indices[0])))
    plt.close(fig)


    file = open("{}/text_retrieval_{}.txt".format(output_folder, str(indices[0]), "w"))

    for i, caps in enumerate(cap_for_image):
        file.write("IMAGE {} \n".format(i))
        for c in caps:
            file.write(c + "\n")
        file.write("\n")
        file.write("CORRECT:  {}".format(captions[indices[i]]))
        file.write("\n \n \n \n ")
    file.close()

def get_data(opt):
    dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)
    file = "{}/data_captions_{}_test.txt".format(dpath, opt.version)

    images = []
    captions = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in reader:
            captions.append(line[1].strip())
            images.append(line[0].strip())
    return images, captions




def t2i(images, captions, caplens, sims, n, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    top_n = {}

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):

        inds = np.argsort(sims[index])[::-1]
        top_n[index] = inds[:n]
    return top_n



def i2t(images, captions, caplens, sims, n, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    top1: highest ranking caption/image for every id
    rank: what is the rank of the corresponding image/caption,
          when 0 the image-caption is matched, because rank is highest
    """
    npts = images.shape[0]
    top_n = {}

    for index in range(npts):
        # sort the array and reverse the order to find most similar caption to image
        inds = np.argsort(sims[index])[::-1]
        top_n[index] = inds[:n]
    return top_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained SCAN model')
    parser.add_argument('--top_n', default=5, type=int, help="number of images to show")
    parser.add_argument('--run', default="Run0", type=str, help='Which run to evaluate')
    parser.add_argument('--checkpoint', default="model_best.pth.tar", type=str, help='which checkpoint to use')
    parser.add_argument('--model_path', default="/$HOME/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--data_path', default="/$HOME/thesis/data/", type=str, help='which checkpoint to use')
    parser.add_argument('--vocab_path', default="/$HOME/thesis/vocab/", type=str, help='which checkpoint to use')
    parser.add_argument('--plot_path', default="/$HOME/runs/", type=str, help='which checkpoint to use')
    parser.add_argument('--change', action='store_true',help='change clothing from all to dresses (trained on all, evaluate on dresses)')
    args = parser.parse_args()
    main(args)
