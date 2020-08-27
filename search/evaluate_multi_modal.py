import sys
sys.path.append("../comb")
sys.path.append("/$HOME/thesis/comb")
import os
import csv
import torch
from vocab import Vocabulary, deserialize_vocab
from model import SCAN
import argparse
from data_ken import get_precomp_loader
from evaluation import encode_data
from search import get_txt_emb, sims_i2i, sims_t2i
import numpy as np
from tqdm import tqdm
import random

"""
evaluate multi-modal search according to papers
"""

def main(args):

    random.seed(17)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "{}/runs/{}/seed1/checkpoint/model_best.pth.tar".format(args.model_folder, args.run)

    # load model and options
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    opt = checkpoint['opt']

    # add because basic is not present in model
    d = vars(opt)
    d['basic'] = False

    # change data path from lisa to current system
    opt.data_path = args.data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab("{}/{}/{}_vocab_{}.json".format(args.vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    img_embs = get_test_emb(opt, vocab, model, device, args.run, args.path_out)

    # load queries
    all_queries = load_queries(opt.data_path, opt.clothing)

    # select random 3167 queries
    queries = random.sample(all_queries, args.nr_queries)

    for query in tqdm(queries):
        query_image_id = int(query[0])
        query_text = query[1]
        target_id = int(query[2])

        query_image_emb = torch.from_numpy(img_embs[query_image_id])

        # get query embedding
        query_text_cap, query_text_cap_len = get_txt_emb(query_text, vocab, model)

        # return similarity measure of SCAN
        s_t2i = sims_t2i(query_text_cap, query_text_cap_len, img_embs, opt, args.similarity, args.scan_sim)

        # return cosine similarity between [0,1]
        s_i2i = sims_i2i(query_image_emb, img_embs, args.similarity)

        sims = (args.alpha * s_i2i) + ((1 - args.alpha) * s_t2i)

        r1, r10, r50 = check_match(sims, target_id, query_image_id)


# find the best matches according to similarity score
def check_match(sims, target_id, query_image_id):
    # set similarity with query image to -inf
    sims[query_image_id] = np.NINF

    # sort array
    inds = np.argsort(sims)[::-1]

    r1 = False
    r10 = False
    r50 = False

    if target_id in inds[:1]:
        r1 = True
        print("epic")

    if target_id in inds[:10]:
        r10 = True
        print("jo")

    if target_id in inds[:50]:
        r50 = True
        print("yes")
    return r1, r10, r50

def load_queries(data_path, clothing):
    dpath = "{}/{}/{}/queries_laenen_1k_test.txt".format(data_path, "Fashion200K_multi", clothing)
    queries = []
    with open(dpath, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for caption in caption_reader:
            queries.append(caption)
    return queries

# get test emebdding
def get_test_emb(opt, vocab, model, device, run, path_out):
    try:
        embs = torch.load("{}/embs/embs_{}.pth.tar".format(path_out, run), map_location=(device))
        print("loading embeddings")
        img_embs = embs["img_embs"]
    except:
        print("Create embeddings")

        if opt.trans:
            dpath = "{}/{}/{}".format(opt.data_path, opt.data_name, opt.clothing)
        else:
            dpath = "{}/{}/{}".format(opt.data_path, "Fashion200K_multi", opt.clothing)

        # get testloader
        test_loader = get_precomp_loader(dpath, "test", vocab, opt,
                            opt.batch_size, False, opt.workers)

        img_embs, cap_embs, cap_lens, freqs = encode_data(model, test_loader)

        if not os.path.exists('{}/embs'.format(path_out)):
            os.makedirs('{}/embs'.format(path_out))

        torch.save({'img_embs': img_embs}, '{}/embs/embs_{}.pth.tar'.format(path_out, run))
        print("Saved embeddings")

    return img_embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal search')
    parser.add_argument('--run', default="fscan2", type=str, help='name of run')
    parser.add_argument('--path_out', default="out_multi_eval", type=str, help='path to plots_scan')
    parser.add_argument('--vocab_path', default="../vocab", type=str, help='path to vocab')
    parser.add_argument('--data_path', default="../data", type=str, help='path to data')
    parser.add_argument('--search', default="multi", type=str, help='What kind of search to perform')
    parser.add_argument('--alpha', default=0.45, type=float, help='How much emphasis on i2i search for multimodal search')
    parser.add_argument('--similarity', default="sum", type=str, help='sum|max|laenen for t2i and i2i')
    parser.add_argument('--scan_sim', default=False, action='store_true', help='For t2i use scan similarity measure of cosine similarity')
    parser.add_argument('--model_folder', default="../comb", type=str, help='path to folder where models are stored')
    parser.add_argument('--nr_queries', default=3167, type=int, help='nr of queries to evaluate')


    args = parser.parse_args()
    main(args)
