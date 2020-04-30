import glob
import os
import json
import random
import numpy as np
import torch
import argparse


def save_hyperparameters(log_path, opt):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open('{}/commandline_args.txt'.format(log_path), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

def find_run_name(opt):
    path = "runs/run*"
    runs = glob.glob(path)
    runs.sort()
    if len(runs) == 0:
        return opt
    elif len(runs) > 0:
        nr_next_run = len(runs) + 1
    opt.model_name = './runs/run{}/checkpoint'.format(nr_next_run)
    opt.logger_name = './runs/run{}/log'.format(nr_next_run)

    return opt

def get_random_indx(nr_examples, max_len):
    rnd = [x for x in range(max_len)]
    random.shuffle(rnd)
    rnd = rnd[:nr_examples]
    return rnd


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_result_subset(rs, r, version, word_asked):
    print("Score {} for word: {}".format(version, word_asked))
    print("r1: %.1f \t (%.1f)" % (rs[0], r[0]))
    print("r5: %.1f \t (%.1f)" % (rs[1], r[1]))
    print("r10: %.1f \t (%.1f)" % (rs[2], r[2]))

def calculate_r(ranks, version):
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)

# count the frequency of the words
def count_words(captions):
    count = {}
    for caption in captions:
        for word in caption.split(" "):
            if word in count.keys():
                count[word] += 1
            else:
                count[word] = 1
    return count

# calculate a frequency score of a caption
def calculatate_freq(captions, count):
    # normalized score for frequency of words in sentence
    freq_score = []
    # frequency of every word in sentence
    freqs = []
    for caption in captions:
        # add zero for <start> sign
        word_count = [0]


        caption = caption.split(" ")
        tot_freq = 0
        caption_l = len(caption)
        for word in caption:
            try:
                tot_freq += count[word]
                word_count.append(count[word])
            except:
                caption_l -= 1
                word_count.append(0)

        freq = tot_freq / caption_l
        freq_score.append(freq)
        word_count.append(0)
        freqs.append(word_count)

    freq_score = normalize(freq_score)
    return freq_score, freqs

def normalize(freq_score):
    max_freq = max(freq_score)
    min_freq = min(freq_score)

    if max_freq == min_freq:
        for i in range(len(freq_score)):
            freq_score[i] = freq_score[i] / min_freq
    else:
        for i in range(len(freq_score)):
            freq_score[i] = (freq_score[i] - min_freq)/(max_freq - min_freq)
    return freq_score

# calculate the adaptive margin according to the frequency scores of captions
def adap_margin(freq_score, scores, margin):
    freq = list(freq_score)
    freq = torch.FloatTensor(freq)
    freq = 0.2 + ( (1- freq) * 0.1)

    margin1 = freq.expand_as(scores)

    margin2 = freq.expand_as(scores).t()

    return margin1, margin2
