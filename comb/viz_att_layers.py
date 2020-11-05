import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vocab import Vocabulary, deserialize_vocab
from model import SCAN, xattn_score_i2t, xattn_score_t2i
from data_ken import PrecompDataset, collate_fn
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

def calculate_attn(dpath, vocab, opt, word, model):
    # load the features
    dset = PrecompDataset(dpath, "test", vocab, opt.version, opt.filter,
                            opt.n_filter, opt.cut, opt.n_cut, opt.txt_enc)

    # filter dataset
    positions = dset.filter_word(word)

    # create dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    total_attn = []
    # collect all attention scores
    with torch.no_grad():

        for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
            # compute the embeddings
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=True)

            if opt.cross_attn == "i2t":
                sim, attn = xattn_score_i2t(img_emb, cap_emb, cap_len, freqs, opt)
            else:
                sim, attn = xattn_score_t2i(img_emb, cap_emb, cap_len, freqs, opt)

            total_attn.append(attn[0][0,:,positions[i]])

        total_attn = torch.stack(total_attn)
        average_attn = torch.mean(total_attn, dim=0)

    return average_attn

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
    parser.add_argument('--run', default="run18", type=str, help='which run')
    parser.add_argument('--out_folder', default="vizAttn", type=str, help='')
    parser.add_argument("--list_words", nargs="+", default=["black", "white", "black", "blue", "green", "red", "floral", "lace", "jersey", "silk", "midi", "sheath"])


    args = parser.parse_args()
    main(args)
