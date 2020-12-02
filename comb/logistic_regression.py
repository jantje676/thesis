import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from vocab import Vocabulary, deserialize_vocab
from model import SCAN
from data_ken import PrecompDataset, PrecompTrans, collate_fn
from evaluation import encode_data
import random
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

"""
Perfrom logistic regression on features from trained models
"""

def main(args):
    random.seed(17)

    min_l = args.min_l
    test_percentage = 0.1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "{}/{}/seed1/checkpoint/model_best.pth.tar".format(args.run_folder, args.run)

    print("LOADING MODEL")
    # load trained SCAN model
    model, opt = load_model(model_path, device)
    model.val_start()


    print("RETRIEVE VOCAB")
    # load vocabulary used by the model
    vocab = deserialize_vocab("{}/{}/{}_vocab_{}.json".format(opt.vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)

    args.list_words = [("black", "white"), ("black", "blue"),
                        ("multicolor", "floral"), ("lace", "jersey"),
                        ("silk", "crepe"), ("maxi", "midi"),
                        ("sheath", "shift"),("sleeve", "sleeveless"),
                        ("long", "knee-length"),("embroidered", "beaded")]
    scores = []
    for pair in args.list_words:
        word1 = pair[0]
        word2 = pair[1]

        dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)

        data_loader_train1, positions_train1 = retrieve_loader("train", opt, dpath, word1, vocab)
        data_loader_train2, positions_train2 = retrieve_loader("train", opt, dpath, word2, vocab)

        features1 = create_embs(data_loader_train1, model)
        features2 = create_embs(data_loader_train2, model)

        f1_best = []
        f1_worst = []
        best_segs = []
        worst_segs = []

        for i in range(5):
            result = perform_exp(model , features1, features2, test_percentage, min_l)
            f1_best.append(result[0])
            f1_worst.append(result[2])
            best_segs.append(result[1])
            worst_segs.append(result[3])
            l = result[4]

        best = np.mean(f1_best)
        worst = np.mean(f1_worst)
        best_std = np.std(f1_best)
        worst_std = np.std(f1_worst)

        best_seg = np.argmax(np.bincount(best_segs))
        worst_seg = np.argmax(np.bincount(worst_segs))

        scores.append((best, best_std, best_seg, best_segs, worst, worst_std, worst_seg, worst_segs, l))

    print_scores(scores, args.list_words)

def perform_exp(model , features1, features2, test_percentage, min_l):
        # create test_set
        x_train, x_test, y_train, y_test, min_l = create_train_test(features1, features2, test_percentage, min_l)

        best_f1 = 0
        best_std = 0
        worst_f1 = 1
        worst_std = 0
        best_hps = None
        best_seg = None
        worst_seg = None

        for i in range(7):
            # choose correct fragment
            x_train_seg = x_train[:, i, :]
            x_test_seg = x_test[:, i, :]

            scaler = StandardScaler()
            scaler.fit(x_train_seg)

            x_train_seg = scaler.transform(x_train_seg)
            x_test_seg = scaler.transform(x_test_seg)

            clf = LogisticRegression(random_state=0).fit(x_train_seg, y_train)
            pred = clf.predict(x_test_seg)
            f1 = calc_score(pred, y_test)

            if f1 > best_f1:
                best_f1 = f1
                # best_hps = hps
                best_seg = i

            if f1 < worst_f1:
                worst_f1 = f1
                worst_seg = i
        return (best_f1, best_seg, worst_f1, worst_seg, min_l * 2)

def print_scores(scores, list_words):
    for i in range(len(scores)):
        word1, word2 = list_words[i]
        score = scores[i]
        print("*********{} vs. {}*********".format(word1, word2))
        print("Best f1-score: {} ({})".format(score[0], score[1]))
        print("Best Segment: {}".format(score[2]))
        print("Best segs: {}".format(score[3]))

        print("Worst f1-score: {} ({})".format(score[4], score[5]))
        print("Worst Segment: {}".format(score[6]))
        print("Worst segs: {}".format(score[7]))

        print("Data size: {}".format(score[8]))

def calc_score(pred, y_test):
    score = (pred == y_test)
    f1 = f1_score(y_test, pred, average='binary')
    return f1

def create_train_test(features1, features2, test_percentage, min_l):
    # find minimum length
    min_nr = min(features1.shape[0], features2.shape[0])


    if min_nr < min_l:
        min_l = min_nr

    print("Total data size is {}".format(min_l * 2))
    range1 = list(range(0, features1.shape[0]))
    range2 = list(range(0, features1.shape[0]))

    # make both sets equal
    indx1 = random.sample(range1, min_l)
    indx2 = random.sample(range2, min_l)

    features1 = np.take(features1, indx1, axis=0)
    features2 = np.take(features2, indx2, axis=0)

    n_test = math.ceil(min_l * test_percentage)
    test1 = features1[:n_test]
    train1 = features1[n_test:]

    test2 = features2[:n_test]
    train2 = features2[n_test:]

    test = np.concatenate((test1, test2), axis=0)
    train = np.concatenate((train1, train2), axis=0)

    y_test = np.concatenate((np.zeros(len(test1)), np.ones(len(test2))), axis=0)
    y_train = np.concatenate((np.zeros(len(train1)),np.ones(len(train2))), axis=0)

    train, y_train = shuffle(train, y_train, random_state=0)
    test, y_test = shuffle(test, y_test, random_state=0)

    return train, test, y_train, y_test, min_l

def create_embs(data_loader, model):

    img_emb, cap_emb, cap_len, _ = encode_data(model, data_loader)
    return img_emb

def retrieve_loader(split, opt, dpath, word, vocab):

    if opt.precomp_enc_type == "trans" or opt.precomp_enc_type == "layers" or opt.precomp_enc_type == "layers_attention" or opt.precomp_enc_type == "cnn_layers" or opt.precomp_enc_type == "layers_attention_res" or opt.precomp_enc_type == "layers_attention_im":
        dset = PrecompTrans(dpath, split, vocab, opt.version, opt.image_path,
                            opt.rectangle, opt.data_name, opt.filter, opt.n_filter,
                            opt.cut, opt.n_cut, opt.clothing, opt.txt_enc)
    else:
        dset = PrecompDataset(dpath, split, vocab, opt.version, opt.filter,
                            opt.n_filter, opt.cut, opt.n_cut, opt.txt_enc)

    # filter dataset
    positions = dset.filter_word(word)

    # create dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader, positions

def load_model(model_path, device):
    # load model and options
    checkpoint = torch.load(model_path, map_location=device)
    opt = checkpoint['opt']

    # add because div_transform is not present in model
    d = vars(opt)
    d['div_transform'] = False

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    return model, opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize attention distribution')
    parser.add_argument('--run_folder', default="runs", type=str, help='path to run folder')
    parser.add_argument('--run', default="run61", type=str, help='which run')
    parser.add_argument("--list_words", nargs="+", default=["black", "white", "black", "blue", "green", "red", "floral", "lace", "jersey", "silk", "midi", "sheath"])
    parser.add_argument('--min_l',help='maximum nr of features for one word', default=400, type=int)


    args = parser.parse_args()
    main(args)
