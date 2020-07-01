# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import csv
from utils import count_words, calculatate_freq, filter_freq, cut
import h5py

class CaptionDataset(data.Dataset):
    """
    Load captions
    """

    def __init__(self, data_path, data_split, vocab, version):
        self.vocab = vocab
        loc = data_path + '/'
        self.captions = []

        with open('{}/data_captions_{}_{}.txt'.format(data_path, version, data_split), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for line in reader:
                self.captions.append(line[1].strip())
        self.length = len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target, index

    def __len__(self):
        return self.length

def collate_fn_caption(data):
    # Sort a data list by caption length
    captions, ids = zip(*data)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return targets, lengths, ids


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, version):
        self.vocab = vocab
        loc = data_path + '/'
        self.captions = []

        with open('{}/data_captions_{}_{}.txt'.format(data_path, version, data_split), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for line in reader:
                self.captions.append(line[1].strip())

        # Image features
        self.images = np.load("{}/data_ims_{}_{}.npy".format(data_path, version, data_split))
        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=False, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dset = PrecompDataset(data_path, data_split, vocab, opt.version)


    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    return data_loader

def get_caption_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=False, num_workers=2):

    dset = CaptionDataset(data_path, data_split, vocab, opt.version)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,

                                              collate_fn=collate_fn_caption)
    return data_loader

def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name, opt.clothing)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      1, False, workers)

    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)

    train_cap_loader = get_caption_loader(dpath, 'train', vocab, opt,
                                      batch_size, False, workers)


    # val_cap_loader = get_caption_loader(dpath, 'dev', vocab, opt,
    #                                 batch_size, False, workers)

    return train_loader, val_loader, train_cap_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name, opt.clothing)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)

    return test_loader
