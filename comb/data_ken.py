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



class PrecompTrans(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, version, image_path, rectangle, data_name, filter, n_filter, cut, n_cut, clothing):
        self.vocab = vocab
        loc = data_path + '/'
        self.captions = []
        self.images = []
        self.data_name = data_name
        self.image_path = image_path
        self.data_path = data_path
        self.filter = filter
        self.n_filter = n_filter
        self.cut = cut
        self.n_cut = n_cut
        self.clothing = clothing

        with open('{}/data_captions_{}_{}.txt'.format(data_path, version, data_split), 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for line in reader:
                self.captions.append(line[1].strip())
                self.images.append(line[0].strip())

        self.h5_images =  get_h5_images(self.data_name, data_split, data_path)

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        self.count  = vocab.count
        freq_score, freqs = calculatate_freq(self.captions, self.count)
        self.freq_score = freq_score
        self.freqs = freqs

        if rectangle:
            self.height = 512
        else:
            self.height = 256


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        img_id = self.images[img_id]

        if self.clothing == "multi":
            # work around to get multi away from the path
            new_path = self.data_path[:-6]
            image = Image.open("{}/{}".format(new_path, img_id))
        elif self.data_name == "Fashion200K":
            # load image
            image = Image.open("{}/{}/{}_0.jpeg".format(self.data_path, self.image_path, img_id))
        elif self.data_name == "Fashion_gen":
            image = self.h5_images[int(img_id)]
            image = Image.fromarray(image)

        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.Resize((self.height, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        image = transform(image)

        caption = self.captions[index]
        vocab = self.vocab
        freq_score = self.freq_score[index]
        freqs = self.freqs[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        if self.filter:
            tokens = filter_freq(tokens, self.count, self.n_filter)

        if self.cut:
            tokens = cut(tokens, self.n_cut)

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, img_id, freq_score, freqs

    def __len__(self):
        return self.length

#
def get_h5_images(data_name, data_split, data_path):
    if data_name == "Fashion200K":
        return None
    elif data_name == "Fashion_gen":
        if data_split == "train":
            file = "{}/fashiongen_256_256_train.h5".format(data_path)
        else:
            file = "{}/fashiongen_256_256_validation.h5".format(data_path)
        f = h5py.File(file, 'r')
        dset = f["input_image"]
        return dset



class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, version, filter, n_filter, cut, n_cut):
        self.vocab = vocab
        loc = data_path + '/'
        self.captions = []
        self.filter = filter
        self.n_filter = n_filter
        self.cut = cut
        self.n_cut = n_cut

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


        self.count = vocab.count
        freq_score, freqs = calculatate_freq(self.captions, self.count)
        self.freq_score = freq_score
        self.freqs = freqs



    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab
        freq_score = self.freq_score[index]
        freqs = self.freqs[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        if self.filter:
            tokens = filter_freq(tokens, self.count, self.n_filter)

        if self.cut:
            tokens = cut(tokens, self.n_cut)

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id, freq_score, freqs

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
    images, captions, ids, img_ids, freq_score, freqs = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, freq_score, freqs


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if opt.trans or opt.precomp_enc_type == "layers" or opt.precomp_enc_type == "layers_attention":
        dset = PrecompTrans(data_path, data_split, vocab, opt.version, opt.image_path,
                            opt.rectangle, opt.data_name, opt.filter, opt.n_filter, opt.cut, opt.n_cut, opt.clothing)
    else:
        dset = PrecompDataset(data_path, data_split, vocab, opt.version, opt.filter,
                                opt.n_filter, opt.cut, opt.n_cut)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name, opt.clothing)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)

    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name, opt.clothing)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
