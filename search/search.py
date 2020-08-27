import matplotlib
import sys

matplotlib.use('Agg')
sys.path.append("../comb")
sys.path.append("/home/kgoei/thesis/comb")

import matplotlib.image as mpimg
import numpy as np
import argparse
import base64
import nltk
import csv
import torchvision.models as models
import torch
import glob
import torch.nn as nn
import os

from utils_search import str2bool, norm, norm_t2i
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from model import SCAN
from evaluation import shard_xattn_t2i
from visualize_attention import get_embs
from torch.autograd import Variable
from vocab import Vocabulary, deserialize_vocab
from util.segment_dresses import segment_dresses

"""
Perform multi-modal search with atribute manipulation
Give a image and text query and find the best matching image in the test set.
NOTE: test set is sometimes only of size 500, is this enough for good retrieval?
"""

def main(args):

    model_path = "{}/runs/{}/seed1/checkpoint/model_best.pth.tar".format(args.model_folder, args.run)

    # load model and options
    checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
    opt = checkpoint['opt']

    # add because basic is not present in model
    d = vars(opt)
    d['basic'] = False

    text_query = args.text_query
    name_image = args.name_image
    path_test_images = args.path_test_images
    run = args.run
    data_path = args.data_path
    n_top = args.n_top
    search = args.search
    alpha = args.alpha
    similarity = args.similarity
    scan_sim = args.scan_sim


    if data_path is not None:
        opt.data_path = data_path

    caption_test_path = "{}/Fashion200K/{}/data_captions_{}_test.txt".format(data_path, opt.clothing, opt.version)
    path_test_image = "{}/{}".format(path_test_images, name_image)
    output_folder = "search_{}_{}".format(run, opt.version)
    vocab_path = "{}/{}/{}_vocab_{}.json".format(args.vocab_path, opt.clothing, opt.data_name, opt.version )

    opt.vocab_path = vocab_path
    version = opt.version

    if opt.trans:
        plot_folder = "plots_trans"
    else:
        plot_folder = "plots_scan"

    plot_path = "../comb/{}".format(plot_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    # load vocabulary
    vocab = deserialize_vocab(vocab_path)

    # get test embeddings
    img_embs, cap_embs, cap_lens = get_test_emb(plot_path, run, version, opt, model, data_path, plot_folder)

    # get query embedding
    query_text_cap, query_text_cap_len = get_txt_emb(text_query, vocab, model)

    # segment image and get features
    query_image_feature = get_img_features(path_test_image, opt)

    # tranform features to embeddings
    query_image_emb = get_img_emb(query_image_feature, model)

    # return similarity measure of SCAN
    s_t2i = sims_t2i(query_text_cap, query_text_cap_len, img_embs, opt, similarity, scan_sim)

    # return cosine similarity between [0,1]
    s_i2i = sims_i2i(query_image_emb, img_embs, similarity)


    if search == "t2i":
        sims = s_t2i
    elif search == "i2i":
        sims = s_i2i
    elif search == "multi":
        sims = (alpha * s_i2i) + ((1 - alpha) * s_t2i)

    # find best matches according to similarity scores
    top_indx = best_matches(sims, n_top)

    retrieve_images(top_indx, caption_test_path, text_query, scan_sim, similarity, search, alpha, path_test_image, name_image, opt, output_folder)


def sims_i2i(query_image_emb, img_embs, similarity):
    if similarity == "max":
        sims = i2i_max(query_image_emb, img_embs)
    elif similarity == "sum" or similarity == "laenen":
        sims = i2i_sum(query_image_emb, img_embs)
    sims = norm(sims)
    return sims

# calculate similarities between text query and images in test set
def sims_t2i(query_text_cap, query_text_cap_len, img_embs, opt, similarity, scan_sim):
    query_text_cap = query_text_cap.data.cpu().numpy()
    query_text_cap_len = [int(query_text_cap_len.double().item())]

    if scan_sim:
        sims, _ = shard_xattn_t2i(img_embs, query_text_cap, query_text_cap_len, opt, shard_size=128)
        sims = np.squeeze(sims)

        #find where similarity score is below zero, remove because it can influence the multi search
        temp = np.argwhere(sims < 0)
        # normalize scores for multi
        sims_norm = norm(sims)
        # replace negative similarities with infinity
        sims_norm[temp] = np.NINF
    else:
        # remove first and last word (<start>, <end>)
        query_text_cap = query_text_cap[: , 1:-1, :]
        temp = []
        cos = nn.CosineSimilarity(dim=2)

        for i in range(query_text_cap.shape[1]):
            query_word = query_text_cap[:, i, :]
            query_word = np.tile(query_word, (img_embs.shape[0], img_embs.shape[1], 1))

            sims_word = cos(torch.from_numpy(query_word), torch.from_numpy(img_embs))
            sims_word = sims_word.data.numpy()

            if similarity == "max":
                sims_word = np.max(sims_word, axis=1)
            elif similarity == "sum":
                sims_word = np.sum(sims_word, axis=1)
            elif similarity == "laenen":
                sims_word = np.where(sims_word > 0 , sims_word, 0)
                sims_word = np.sum(sims_word, axis=1)
                sims_word = np.where(sims_word > 0 , sims_word, np.NINF)
            temp.append(sims_word)

        sims = np.stack(temp, axis=1)
        sims = np.sum(sims, axis=1)

        sims_norm = norm_t2i(sims)

    return sims_norm

def i2i_max(query_image_emb, img_embs):
    img_embs = torch.from_numpy(img_embs).float()

    # numerator
    sims_embs = torch.einsum('ik,ljk->lij', query_image_emb, img_embs)

    # calulcate norms from both tensors
    query_norm = torch.norm(query_image_emb, dim= 1)
    im_norm = torch.norm(img_embs, dim= 2)

    # denominator
    norm_sims = torch.einsum('i,lj->lij', query_norm, im_norm)

    # calcualte simalirity between all image fragments
    sims = sims_embs / norm_sims
    sims = torch.max(sims, dim=2)[0]
    sims = torch.sum(sims, dim=1)

    return sims.data.numpy()

def i2i_sum(query_image_emb, img_embs):
    if len(query_image_emb.shape) != 3:
        query_image_emb = query_image_emb.unsqueeze(0)

    # expand tensor for cosine similarity
    repeat_query_image_emb = query_image_emb.expand(img_embs.shape[0], -1, -1)

    cos = nn.CosineSimilarity(dim=2)
    sims = cos(repeat_query_image_emb, torch.from_numpy(img_embs))

    sims = sims.data.numpy()
    sims = np.sum(sims, axis=1)
    return sims

def retrieve_images(top_indx, caption_test_path, text_query, scan_sim, similarity, search, alpha, path_test_image, name_image, opt, output_folder):
    test_id2data = {}
    # find the caption and image with every id in the test file {caption_id : (image_id, caption)}
    with open(caption_test_path, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for i, line in enumerate(caption_reader):
            test_id2data[i] = (line[0], line[1])


    img_adresses = []
    titles = []
    if search == "multi" or search =="i2i":
        img_adresses.append(path_test_image)
        titles.append("Query image")

    for indx in top_indx:
        img_adress = "../data/{}/{}/pictures_only/pictures_only/{}_0.jpeg".format(opt.data_name, opt.clothing, test_id2data[indx][0])
        img_adresses.append(img_adress)

        titles.append(test_id2data[indx][1])

    fig = plt.figure(figsize=(40, 20))
    fig.tight_layout()

    columns = len(img_adresses)
    rows = 1
    ax = []

    for j,img_adress in enumerate(img_adresses):
        img = mpimg.imread(img_adress)

        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, j+1))
        ax[-1].set_title(titles[j], fontsize=20)
        plt.imshow(img)

    if search == "multi" or search =="t2i":
        fig.suptitle("Query text: {}".format(text_query), fontsize=20)
    if scan_sim and search != "multi":
        plt.savefig('{}/{}_scan_sim_{}_{}.png'.format(output_folder, search, text_query, name_image))
    elif scan_sim and search == "multi":
        plt.savefig('{}/{}_{}_scan_sim_{}_{}.png'.format(output_folder, search,alpha, text_query, name_image))
    elif search == "multi":
        plt.savefig('{}/{}_{}_cosine_{}_{}_{}.png'.format(output_folder, search, alpha, text_query, similarity, name_image))
    else:
        plt.savefig('{}/{}_cosine_{}_{}_{}.png'.format(output_folder, search, text_query, similarity, name_image))
    plt.close(fig)


# find the best matches according to similarity score
def best_matches(sims, n_top):
    # sort array
    inds = np.argsort(sims)[::-1][:n_top]
    return inds


# create image embedding
def get_img_emb(query_image_feature, model):
    query_image_feature = torch.Tensor(query_image_feature)
    query_image_feature = Variable(query_image_feature, volatile=False)
    img_emb = model.img_enc(query_image_feature)
    return img_emb

# get test emebdding
def get_test_emb(plots_scan_path, run, version, opt, model, data_path, plot_path):
    try:
        embs = torch.load("{}/embs/embs_{}_{}.pth.tar".format(plots_scan_path, run, version), map_location=('cpu'))
        print("loading embeddings")
        img_embs = embs["img_embs"]
        cap_embs = embs["cap_embs"]
        cap_lens = embs["cap_lens"]
    except:
        print("Create embeddings")
        img_embs, cap_embs, cap_lens, _ = get_embs(opt, model ,run , version, data_path, plot_path, vocab_path="../vocab/{}".format(opt.clothing))

    return img_embs, cap_embs, cap_lens

# get the text embeddings for the query
def get_txt_emb(text_query, vocab, model):
    tokens = nltk.tokenize.word_tokenize(str(text_query).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption).long()

    target = Variable(target, volatile=False)
    target = target.unsqueeze(0)

    cap_l = [int(len(caption))]

    if torch.cuda.is_available():
        target = target.cuda()
        cap_l = target.cuda()
    cap_emb, cap_lens = model.txt_enc(target, cap_l)

    return cap_emb, cap_lens

# create image features with a pretrained neural net
def get_img_features(path_test_image, opt):
    if opt.trans and opt.rectangle:
        height = 512
    else:
        height = 256

    transform = transforms.Compose([
        transforms.Resize((height, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    if opt.trans:
        image = Image.open(path_test_image)
        features = transform(image).unsqueeze(0)
    else:
        # choose model
        net = models.alexnet(pretrained=True)
        # take aways the last layers
        net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])

        # set to evaluation
        net.eval()

        img = mpimg.imread(path_test_image)
        segments, _ = segment_dresses(img)

        features = []

        # push segments through the net
        for key in segments:
            seg_pil = Image.fromarray(segments[key])

            # transform images
            seg_transformed = transform(seg_pil)

            # add extra dimension
            seg_transformed = seg_transformed.unsqueeze(0)

            feature = net(seg_transformed)

            feature = feature.squeeze()
            features.append(feature.detach().numpy())
        features = np.stack(features, axis=0)
    return features



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal search')
    parser.add_argument('--text_query', default="blue", type=str, help='text query for search')
    parser.add_argument('--name_image', default="test6.jpg", type=str, help='name of the query image in the test images folder')
    parser.add_argument('--path_test_images', default="test_images", type=str, help='name folder to store query images')
    parser.add_argument('--run', default="run57", type=str, help='name of run')
    parser.add_argument('--plots_scan_path', default="../comb/plots_scan", type=str, help='path to plots_scan')
    parser.add_argument('--vocab_path', default="../vocab", type=str, help='path to vocab')
    parser.add_argument('--data_path', default="../data/", type=str, help='path to data')
    parser.add_argument('--n_top', default=5, type=int, help='top n examples to be returned by search')
    parser.add_argument('--search', default="multi", type=str, help='What kind of search to perform')
    parser.add_argument('--alpha', default=0.45, type=float, help='How much emphasis on i2i search for multimodal search')
    parser.add_argument('--similarity', default="sum", type=str, help='sum|max|laenen for t2i and i2i')
    parser.add_argument('--scan_sim', default=False, type=str2bool, help='For t2i use scan similarity measure of cosine similarity')
    parser.add_argument('--model_folder', default="../comb", type=str, help='path to folder where models are stored')

    args = parser.parse_args()
    main(args)
