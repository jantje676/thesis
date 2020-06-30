from train import start_experiment
import argparse
from utils import find_run_name, set_run_name
import tb as tb_logger
import logging
import time
import os
import csv
import numpy as np
from random import Random

def main(opt):
    nr_runs = opt.nr_runs
    seeds = 17
    randomHyper = Random()
    output_folder = "tuning_hyper"
    # find run name and set to seed1
    nr_next_run = find_run_name(opt)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open('{}/results_tuning_{}.txt'.format(output_folder, time.time()), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(("Run", "Margin", "word_dim", "embed_size", "n_layers", "lr",
                        "lr_update","raw_feature_n", "agg_func", "cross_attn", "precomp",
                        "lambda_lse", "lambda_softmax", "no_imgn", "max_vio", "no_txtn", "Score"))


        for i in range(nr_runs):

            opt = set_run_name(opt, nr_next_run, i+1)
            opt = random_params(opt, randomHyper)


            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            tb_logger.configure(opt.logger_name, flush_secs=5)

            rsum = start_experiment(opt, seeds)


            writer.writerow((i, "%.2f" % opt.margin, opt.word_dim,opt.embed_size, opt.num_layers,
                                opt.learning_rate, opt.lr_update, opt.raw_feature_norm, opt.agg_func, opt.cross_attn,
                                opt.precomp_enc_type, opt.lambda_lse, opt.lambda_softmax, opt.no_imgnorm, opt.max_violation, opt.no_txtnorm,rsum))

def random_params(opt, randomHyper):
    opt.margin = randomHyper.random()
    opt.word_dim = randomHyper.choice ([200, 300, 400])
    opt.embed_size = randomHyper.choice ([512, 1024, 1536])
    opt.num_layers = randomHyper.choice ([ 1,2,3])
    opt.learning_rate= randomHyper.choice ([0.002, 0.0002, 0.00002, 0.000002])
    opt.lr_update = randomHyper.choice ([5, 15, 25, 35])
    opt.raw_feature_norm = randomHyper.choice (["clipped_l2norm", "l2norm", "no_norm", "softmax"])
    opt.agg_func = randomHyper.choice (["LogSumExp","Mean","Max","Sum"])
    opt.cross_attn = randomHyper.choice (["i2t", "t2i"])
    opt.precomp_enc_type = randomHyper.choice (["basic","weight_norm"])
    opt.lambda_lse = float(randomHyper.randint(1,10))
    opt.lambda_softmax = float(randomHyper.randint(1,15))
    opt.no_imgnorm = randomHyper.choice ([True, False])
    opt.max_violation = randomHyper.choice ([True, False])
    opt.no_txtnorm = randomHyper.choice ([True, False])

    return opt


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data',
                        help='path to datasets')
    parser.add_argument('--clothing', default='dresses',
                        help='which clothing item')
    parser.add_argument('--data_name', default='Fashion200K',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='../vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/run0/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/run0/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum|Freq')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm|attention')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--version', default="laenen", type=str,
                        help='version.')
    parser.add_argument('--adap_margin', action='store_true',
                        help="use adaptive margin")
    parser.add_argument('--add_cost', action='store_true',
                        help="use extra cost for low frequency sentences")
    parser.add_argument('--cost_thres', default=0.4, type=float,
                        help='threhold use for cost function')
    parser.add_argument('--gamma', default=0.8, type=float,
                        help='(Used for add_cost) fraction of normal similarity used for sentences that have low frequency')
    parser.add_argument('--epsilon', default=1.0, type=float,
                        help='(Used for agg_func=frew) regulaizer for emphasis on non-frequent words')
    parser.add_argument('--nr_runs', default=1, type=int,
                        help='Number of experiments.')
    parser.add_argument('--n_attention', default=7, type=int,
                        help='Number of attention maps to create.')
    parser.add_argument('--seed1', default=17, type=int,
                        help='first seed to change easily')

    # filter options for descriptions fashion-Gn
    parser.add_argument('--filter', action='store_true',
                        help="filter descriptions for most frequent words")
    parser.add_argument('--n_filter', default=8, type=int,
                        help='filter for most n most frequent words')
    parser.add_argument('--cut', action='store_true',
                        help="cut descriptions after n_words")
    parser.add_argument('--n_cut', default=5, type=int,
                        help='cut descriptions after n words')


    # Specific for trans
    parser.add_argument('--trans', action='store_true',
                        help="use SCAN with spatial transformers")
    parser.add_argument('--image_path', default="pictures_only/pictures_only", type=str,
                        help='Number of experiments.')
    parser.add_argument('--n_detectors', default=5, type=int,
                        help='How many localizers to use for spatial transformers')
    parser.add_argument('--pretrained_alex', action='store_true',
                        help="use pretrained alexnets for features extractors")
    parser.add_argument('--rectangle', action='store_true',
                        help="use rectangle ratio for images")
    parser.add_argument('--shard_size', default=128, type=int,
                        help='batch size for validating')

    opt = parser.parse_args()
    main(opt)
