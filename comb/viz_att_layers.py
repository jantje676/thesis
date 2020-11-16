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

"""
Create viz for layers experiment given a word, show distribution attention over the different layers
"""

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "{}/{}/seed1/checkpoint/model_best.pth.tar".format(args.run_folder, args.run)
    out_path = "{}/{}".format(args.out_folder, args.run)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load trained SCAN model
    model, opt = load_model(model_path, device)
    model.val_start()


    # load vocabulary used by the model
    vocab = deserialize_vocab("{}/{}/{}_vocab_{}.json".format(opt.vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)

    word_attn = {}
    for word in args.list_words:
        dpath = os.path.join(opt.data_path, opt.data_name, opt.clothing)
        try:
            average_attn = calculate_attn(dpath, vocab, opt, word, model)
        except:
            print("Word ({}) not found".format(word))
            continue
        plot_attn(average_attn, out_path, word)
        word_attn[word] = average_attn

    write_attn(out_path, word_attn)
    plot_one(out_path, word_attn)

def plot_one(out_path, word_attn):
    words = ['black', 'green', 'red', 'white', 'jersey', 'medi', 'sheath', "lace"]

    x = 2
    y = int(len(words)/x)
    fig, axs = plt.subplots(x, y, figsize=(40,15))

    for i in range(x):
        for j in range(y):
            word = words[(i*y) + j]
            average_attn = word_attn[word]
            n_layers = torch.tensor(average_attn.shape)[0].item()
            x_axes = list(range(1, n_layers + 1))
            axs[i, j].bar(x_axes, average_attn.cpu(), align='center', alpha=0.5)
            axs[i, j].set_title(word)
            axs[i, j].set_xlabel("layer")
            axs[i, j].set_ylabel("attention")

    plt.savefig('{}/viz_one'.format(out_path))
    plt.close()

    return
def write_attn(out_path, word_attn):
    file = open("{}/attention.txt".format(out_path), "w")
    for key in word_attn.keys():
        file.write(str(key) + "\t" + str(word_attn[key]) + "\n")
    file.close()

def plot_attn(average_attn, out_path, word):
    n_layers = torch.tensor(average_attn.shape)[0].item()
    x = list(range(1, n_layers + 1))
    plt.bar(x, average_attn.cpu(), align='center', alpha=0.5)
    plt.xlabel('Layer')
    plt.ylabel('Attention')
    plt.title(word)
    plt.savefig('{}/viz_{}'.format(out_path, word))
    plt.close()

def calculate_attn(dpath, vocab, opt, word, model):
    # load the features

    data_loader_test, positions_test = retrieve_loader("test", opt, dpath, word, vocab)
    data_loader_train, positions_train = retrieve_loader("train", opt, dpath, word, vocab)

    total_attn = []
    total_attn += create_attn(data_loader_test, positions_test, opt, model)
    total_attn += create_attn(data_loader_train, positions_train, opt, model)

    total_attn = torch.stack(total_attn)
    average_attn = torch.mean(total_attn, dim=0)

    return average_attn

def create_attn(data_loader, positions, opt, model):
    # collect all attention scores
    total_attn = []

    with torch.no_grad():

        for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
            # compute the embeddings
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=True)

            if opt.cross_attn == "i2t":
                sim, attn = xattn_score_i2t(img_emb, cap_emb, cap_len, freqs, opt)
            else:
                sim, attn = xattn_score_t2i(img_emb, cap_emb, cap_len, freqs, opt)

            total_attn.append(attn[0][0,:,positions[i]])
    return total_attn

def retrieve_loader(split, opt, dpath, word, vocab):

    if opt.precomp_enc_type == "trans" or opt.precomp_enc_type == "layers" or opt.precomp_enc_type == "layers_attention" or opt.precomp_enc_type == "cnn_layers":
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
