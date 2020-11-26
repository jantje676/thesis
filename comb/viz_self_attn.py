import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vocab import Vocabulary, deserialize_vocab
from model import SCAN, xattn_score_i2t, xattn_score_t2i
from data_ken import PrecompDataset, PrecompTrans, collate_fn
from evaluation import encode_data
import math
"""
Create viz for self attention in layers_attenion
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

    print("FILTER DATASETS")
    word_attn = {}
    for word in args.list_words:
        dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)
        # try:
        average_attn = calculate_attn(dpath, vocab, opt, word, model)
        # except:
        #     print("Word ({}) not found".format(word))
        #     continue
        plot_one(out_path, average_attn, word)

def plot_one(out_path, attn, word):
    n_feat = attn.shape[0]
    n_layers = attn.shape[1]
    x = n_feat

    fig, axs = plt.subplots(x,figsize=(40,40))

    for i in range(n_feat):

        average_attn = attn[i]
        x_axes = list(range(1, n_layers + 1))
        axs[i].bar(x_axes, average_attn.cpu(), align='center', alpha=0.5)
        axs[i].set_title("feature {}: {}".format(i+1, word))
        axs[i].set_xticks(x_axes)
        axs[i].set_xlabel("layer")
        axs[i].set_ylabel("attention")

    plt.savefig('{}/viz_one_SELF_{}'.format(out_path, word))
    plt.close()

    return

def calculate_attn(dpath, vocab, opt, word, model):
    # load the features

    data_loader_test, positions_test = retrieve_loader("test", opt, dpath, word, vocab)
    data_loader_train, positions_train = retrieve_loader("train", opt, dpath, word, vocab)

    total_attn = []
    total_attn += create_attn(data_loader_test, positions_test, opt, model)
    total_attn += create_attn(data_loader_train, positions_train, opt, model)

    total_attn = torch.cat(total_attn, dim=0)
    average_attn = torch.mean(total_attn, dim=0)
    return average_attn

def create_attn(data_loader, positions, opt, model):
    # collect all attention scores
    total_attn = []

    with torch.no_grad():

        for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
            # compute the embeddings
            img_emb, cap_emb, cap_len, attn = model.forward_emb_attention(images, captions, lengths, volatile=True)

            total_attn.append(attn)
    return total_attn

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
    parser.add_argument('--run', default="run19", type=str, help='which run')
    parser.add_argument('--out_folder', default="vizAttn", type=str, help='')
    parser.add_argument("--list_words", nargs="+", default=["black", "white", "black", "blue", "green", "red", "floral", "lace", "jersey", "silk", "midi", "sheath"])


    args = parser.parse_args()
    main(args)
