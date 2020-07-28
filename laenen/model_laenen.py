# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from utils import adap_margin

from LaenenLoss import LaenenLoss

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    img_enc = EncoderImagePrecomp(img_dim, embed_size)
    return img_enc



class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted = False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        return cap_emb, cap_len





def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    temp = (w12 / (w1 * w2).clamp(min=eps))

    return temp.squeeze()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
            cudnn.enabled = True

        self.criterion = LaenenLoss(opt.margin, opt.n, opt.switch, opt.beta, opt.gamma)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.SGD(params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.alpha)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_emb_captions(self, captions, lengths):
        """Compute the caption embeddings
        """

        if torch.cuda.is_available():
            captions = captions.cuda()

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return cap_emb, cap_lens

    def forward_loss(self, epoch, img_emb_pair, cap_emb, cap_l, image_diag, cap_diag, same, kmeans_features, kmeans_emb, features, cluster_loss):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(epoch, img_emb_pair, cap_emb, cap_l, image_diag, cap_diag, same, kmeans_features, kmeans_emb, features, cluster_loss)
        self.logger.update('Le', loss.item(), img_emb_pair.size(0))
        return loss


    def train_emb(self, epoch, first_data, second_data, same, cluster_loss, kmeans_features, kmeans_emb):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # unpack train data
        im_first, cap_first, lengths_first, ids_first = first_data
        # unpack pair data
        im_second, cap_second, lengths_second, ids_second = second_data

        # retrieve embeddings from pair data
        img_emb_first, cap_emb_first, l_first = self.forward_emb(im_first, cap_first, lengths_first)

        img_emb_second, cap_emb_second, l_second = self.forward_emb(im_second, cap_second, lengths_second)

        # calculate the similairty score between the image and caption pair
        image_diag = self.criterion.sim_pair(img_emb_first, cap_emb_first, l_first)

        cap_diag = self.criterion.sim_pair(img_emb_second, cap_emb_second, l_second)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_emb_first, cap_emb_second, l_second,
                                image_diag, cap_diag, same, kmeans_features,
                                kmeans_emb, im_first, cluster_loss)

        # compute gradient and do SGD step
        loss.backward()

        self.optimizer.step()
