import os
import argparse
import torch
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vocab import Vocabulary, deserialize_vocab
from model import SCAN, xattn_score_i2t, xattn_score_t2i_cosine, cosine_similarity
from data_ken import PrecompDataset, PrecompTrans, collate_fn
from evaluation import encode_data
import pandas as pd
import seaborn as sns
"""
Script to calculate the cosine similarity between word and average feature representation from attention
"""

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "{}/{}/seed1/checkpoint/model_best.pth.tar".format(args.run_folder, args.run)
    out_path = "{}/{}".format(args.out_folder, args.run)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    print("LOADING MODEL")
    # load trained SCAN model
    model, opt = load_model(model_path, device)
    model.val_start()


    print("RETRIEVE VOCAB")
    # load vocabulary used by the model
    vocab = deserialize_vocab("{}/{}/{}_vocab_{}.json".format(opt.vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)

    word_cos = {}
    for word_row in args.list_words:
        dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)

        loader_test, pos_test = retrieve_loader("test", opt, dpath, word_row, vocab)
        loader_train, pos_train = retrieve_loader("train", opt, dpath, word_row, vocab)

        img_features_train = create_attn(loader_train, pos_train, opt, model)
        img_features_test = create_attn(loader_test, pos_test, opt, model)

        img_features = torch.cat((img_features_test, img_features_train), dim=0)

        n_image = img_features.shape[0]

        temp_cos = {}
        for word_col in args.list_words:
            word_feature = avg_features_word(word_col, model, vocab)
            word_features = word_feature.expand(n_image,-1)
            cosine_scores = cosine_similarity(word_features, img_features)
            temp_cos[word_col] = torch.mean(cosine_scores).item()

        word_cos[word_row] = temp_cos


    print("PLOT ATTENTION")
    write_table(out_path, word_cos)
    write_fig(out_path, word_cos, args.run)


def avg_features_word(word, model, vocab):
    with torch.no_grad():
        caption = []
        caption.append(vocab('<start>'))
        caption.append(vocab(word))
        caption.append(vocab('<end>'))

        word_vec = torch.Tensor(caption).unsqueeze(dim=0)
        word_vec = word_vec.long()
        length = torch.tensor([3])

        if torch.cuda.is_available():
            word_vec = word_vec.cuda()


        cap_emb, cap_lens = model.txt_enc(word_vec, length)
        word_feature = cap_emb.squeeze()[1]

    return word_feature

def write_out(out_path, dic, file_name):
    file = open("{}/{}.txt".format(out_path, file_name), "w")
    for key in dic.keys():
        file.write(str(key) + "\t" + str(dic[key]) + "\n")
    file.close()

def write_fig(out_path, scores, run):
    df = pd.DataFrame(scores)
    plt.figure(figsize=(15,15))
    plot = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=-1, vmax=1)
    plot.set(xlabel='image_features', ylabel='text_features', title=run)

    fig = plot.get_figure()
    fig.savefig("{}/cosine_figure_laenen.png".format(out_path))


def write_table(out_path, scores):
    file = open("{}/cosine_table_laenen.txt".format(out_path), "w")
    file.write("\t")

    for key in scores.keys():
        file.write(str(key) + "\t")
    file.write("\n")


    for key in scores.keys():
        file.write(str(key) + "\t")
        for row_key in scores[key].keys():
            file.write("{:.3f} \t".format(scores[key][row_key]))
        file.write("\n")

    file.close()



def create_attn(data_loader, positions, opt, model):
    # collect all attention scores
    img_features = []

    with torch.no_grad():

        for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
            # compute the embeddings
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=True)

            if opt.cross_attn == "i2t":
                sim, attn = xattn_score_i2t(img_emb, cap_emb, cap_len, freqs, opt)
            else:
                row_sim = xattn_score_t2i_cosine(img_emb, cap_emb, cap_len, freqs, opt).squeeze()

            img_features.append(row_sim[positions[i]])
    features = torch.stack(img_features, dim=0)

    return features

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
    # d = vars(opt)
    # d["layernorm"] = False
    # d['div_transform'] = False
    # d["net"] = "alex"
    # d["txt_enc"] = "basic"
    # d["diversity_loss"] = None

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    return model, opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize attention distribution')
    parser.add_argument('--run_folder', default="runs", type=str, help='path to run folder')
    parser.add_argument('--run', default="run15", type=str, help='which run')
    parser.add_argument('--out_folder', default="vizAttn", type=str, help='')
    parser.add_argument("--list_words", nargs="+", default=["black", "blue", "white", "red","multicolor","floral", "sheath", "midi", "maxi", "short", "knee-length", "crepe", "v-neck", "jersey", "lace", "silk", "cotton"])


    args = parser.parse_args()
    main(args)
