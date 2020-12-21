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
from transformers import BertTokenizer



class PrecompTrans(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, version, image_path, rectangle, data_name, filter, n_filter, cut, n_cut, clothing, txt_enc):
        self.bert = True if txt_enc == "bert" else False
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


        self.im_div = 1

        self.count  = vocab.count
        freq_score, freqs = calculatate_freq(self.captions, self.count)
        self.freq_score = freq_score
        self.freqs = freqs
        self.height = 512 if rectangle else 256

        if self.bert == True:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = nltk.tokenize



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
            transforms.Resize((256, 256)),
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

        if self.bert:
            self.bert_tokenize(caption)
        else:
            self.normal_tokenize(caption)
        return image, target, index, img_id, freq_score, freqs

    def __len__(self):
        return self.length

    def bert_tokenize(self, caption):
        tokenized_cap = self.tokenizer.encode(caption, add_special_tokens=False)
        target = torch.Tensor(tokenized_cap)
        return target

    def normal_tokenize(self,caption):
        # Convert caption (string) to word ids.
        tokens = self.tokenizer.word_tokenize(
            str(caption).lower())

        if self.filter:
            tokens = filter_freq(tokens, self.count, self.n_filter)

        if self.cut:
            tokens = cut(tokens, self.n_cut)
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return target

    # filter dataset to only contain features with certain word for viz_attn_layers
    def filter_word(self, word):
        print("start filtering dataset for: {}".format(word))
        filtered_captions = []
        filtered_images = []
        filtered_freq_score = []
        filtered_freqs = []
        position = []

        if self.bert:
            temp = self.tokenizer.encode(str(word), add_special_tokens=True)
            word = temp[0]

        for i in range(len(self.captions)):
            if self.bert:
                tokens = self.tokenizer.encode(self.captions[i], add_special_tokens=False)
            else:
                tokens = self.tokenizer.word_tokenize(
                    str(self.captions[i]).lower())
            if word in tokens:
                filtered_captions.append(self.captions[i])
                filtered_images.append(self.images[i])
                filtered_freq_score.append(self.freq_score[i])
                filtered_freqs.append(self.freqs[i])
                indx = tokens.index(word)
                # add +1, because special tokens for padding are added while training
                position.append(indx + 1)

        self.length = len(filtered_captions)
        self.captions = filtered_captions
        self.images = filtered_images

        self.freq_score = filtered_freq_score
        self.freqs = filtered_freqs
        print("dataset filtered, word: {} \t size: {}".format(word, self.length))
        return position




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

    def __init__(self, data_path, data_split, vocab, version, filter, n_filter, cut, n_cut, txt_enc):
        self.bert = True if txt_enc == "bert" else False
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
        self.im_div = 1


        self.count = vocab.count
        freq_score, freqs = calculatate_freq(self.captions, self.count)
        self.freq_score = freq_score
        self.freqs = freqs

        if self.bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = nltk.tokenize

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab
        freq_score = self.freq_score[index]
        freqs = self.freqs[index]

        if self.bert:
            target = self.bert_tokenize(caption)
        else:
            target = self.normal_tokenize(caption)


        return image, target, index, img_id, freq_score, freqs

    def __len__(self):
        return self.length


    def bert_tokenize(self, caption):
        tokenized_cap = self.tokenizer.encode(caption, add_special_tokens=True)
        target = torch.Tensor(tokenized_cap)
        return target

    def normal_tokenize(self,caption):
        # Convert caption (string) to word ids.
        tokens = self.tokenizer.word_tokenize(
            str(caption).lower())

        if self.filter:
            tokens = filter_freq(tokens, self.count, self.n_filter)

        if self.cut:
            tokens = cut(tokens, self.n_cut)
        caption = []

        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return target

    # filter dataset to only contain features with certain word for viz_attn_layers
    def filter_word(self, word):
        print("start filtering dataset for: {}".format(word))
        print(self.images.shape)
        filtered_captions = []
        filtered_images = []
        filtered_freq_score = []
        filtered_freqs = []
        position = []

        if self.bert:
            temp = self.tokenizer.encode(str(word), add_special_tokens=False)
            word = temp[0]

        for i in range(len(self.captions)):
            if self.bert:
                tokens = self.tokenizer.encode(self.captions[i], add_special_tokens=False)
            else:
                tokens = self.tokenizer.word_tokenize(
                    str(self.captions[i]).lower())
            if word in tokens:
                filtered_captions.append(self.captions[i])
                filtered_images.append(self.images[i])
                filtered_freq_score.append(self.freq_score[i])
                filtered_freqs.append(self.freqs[i])
                indx = tokens.index(word)
                # add +1, because special tokens for padding are added while training
                position.append(indx + 1)

        self.length = len(filtered_captions)
        self.captions = filtered_captions
        self.images = np.stack(filtered_images, axis=0)


        self.freq_score = filtered_freq_score
        self.freqs = filtered_freqs
        print("dataset filtered, word: {} \t size: {}".format(word, self.length))
        return position


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
    if opt.precomp_enc_type == "trans" or opt.precomp_enc_type == "layers" or opt.precomp_enc_type == "layers_attention" or opt.precomp_enc_type == "cnn_layers" or opt.precomp_enc_type == "layers_attention_res" or opt.precomp_enc_type == "layers_attention_im":
        dset = PrecompTrans(data_path, data_split, vocab, opt.version, opt.image_path,
                            opt.rectangle, opt.data_name, opt.filter, opt.n_filter,
                            opt.cut, opt.n_cut, opt.clothing, opt.txt_enc)
    else:
        dset = PrecompDataset(data_path, data_split, vocab, opt.version, opt.filter,
                                opt.n_filter, opt.cut, opt.n_cut, opt.txt_enc)

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
