import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
from util.layers_alex2 import LayersModelAlex
from util.layers_res2 import LayersModelRes
from util.layers_res3 import LayersModelResDeep
from util.DeepFashion2 import LayersModelAttr

"""Attention module with the use of layernorm and the adding of the global image as described in the paper by Song"""
class LayerAttention3(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False, net='res'):
        super(LayerAttention3, self).__init__()

        if net == 'alex':
            self.layers = LayersModelAlex(img_dim, embed_size)
        elif net == 'res':
            self.layers = LayersModelRes(img_dim, embed_size)
        elif net == 'attr':
            self.layers = LayersModelAttr(img_dim, embed_size)
        elif net == "res_deep":
            self.layers = LayersModelResDeep()
        self.attention = SelfAttention( img_dim, embed_size, n_attention, no_imgnorm)


    def forward(self, images):
        layer_features, global_feature = self.layers(images)
        features = self.attention.forward_glob(layer_features, global_feature)

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

        super(LayerAttention3, self).load_state_dict(new_state)


class SelfAttention(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.w1 = nn.Linear(img_dim, n_attention, bias=False)
        # self.w2 = nn.Linear(int(img_dim/2), n_attention,  bias=False)
        self.w3 = nn.Linear(img_dim, img_dim, bias=True)
        self.w4 = nn.Linear(img_dim, embed_size)
        self.relu = nn.ReLU()
        self.n_attention = n_attention
        self.layerNorm = nn.LayerNorm(img_dim)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        nn.init.xavier_uniform_(self.w1.weight)
        # nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w4.weight)

    def forward_glob(self, images, global_feature):
        """Self attention on features"""
        # assuming that the precomputed features are already l2-normalized
        attention = self.w1(images)
        attention == self.relu(attention)
        # attention = self.w2(attention)
        attention = F.softmax(attention, dim=1)
        attention = attention.transpose(1,2)

        features = torch.bmm(attention, images)
        features = self.w3(features)

        glob = global_feature.unsqueeze(dim=1)
        glob = glob.expand(-1, self.n_attention, -1)


        features = self.layerNorm(glob + features)


        features = self.w4(features)

        # normalize in the joint embedding space
        features = l2norm(features, dim=-1)
        return features



    def forward_attention(self, images):
        """Self attention on features, pass attention for evaluation"""
        # assuming that the precomputed features are already l2-normalized
        attention = self.w1(images)


        attention = F.tanh(attention)
        attention = self.w2(attention)
        attention = F.softmax(attention, dim=1)
        attention = attention.transpose(1,2)

        features = torch.bmm(attention, images)
        features = self.w3(features)

        # normalize in the joint embedding space
        features = l2norm(features, dim=-1)

        return features, attention

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(SelfAttention, self).load_state_dict(new_state)


class MultiheadAttention(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(MultiheadAttention, self).__init__()
        self.multi = nn.MultiheadAttention(img_dim, 4)
        self.attention = SelfAttention(img_dim, embed_size, n_attention)


    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        out = self.multi(images, images, images)
        features = self.attention(out[0])
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

        super(MultiheadAttention, self).load_state_dict(new_state)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
