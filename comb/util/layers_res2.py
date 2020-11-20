import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict

class LayersModelRes(nn.Module):
    def __init__(self, feature_dim=2048, trained_dresses=False, checkpoint_path=None):
        super(LayersModelRes, self).__init__()
        self.feature_dim = feature_dim
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.a = net.conv1
        self.b = nn.Sequential(net.bn1, net.relu, net.maxpool)
        self.c = net.layer1[0]
        self.d = net.layer1[1]
        self.e = net.layer1[2]
        self.f = net.layer2[0]
        self.g = net.layer2[1]
        self.h = net.layer2[2]
        self.i = net.layer2[3]
        self.j = net.layer3[0]
        self.k = net.layer3[1]
        self.l = net.layer3[2]
        self.m = net.layer3[3]
        self.n = net.layer3[4]
        self.o = net.layer3[5]
        self.p = net.layer4[0]
        self.q = net.layer4[1]
        self.r = net.layer4[2]
        self.s = net.avgpool

    def forward(self, x):
        temp = []

        x = self.a(x)
        y = self.flat(x)
        temp.append(y)

        x = self.b(x)
        y = self.flat(x)
        temp.append(y)

        x = self.c(x)
        y = self.flat(x)
        temp.append(y)

        x = self.d(x)
        y = self.flat(x)
        temp.append(y)

        x = self.e(x)
        y = self.flat(x)
        temp.append(y)

        x = self.f(x)
        y = self.flat(x)
        temp.append(y)

        x = self.g(x)
        y = self.flat(x)
        temp.append(y)

        x = self.h(x)
        y = self.flat(x)
        temp.append(y)

        x = self.i(x)
        y = self.flat(x)
        temp.append(y)

        x = self.j(x)
        y = self.flat(x)
        temp.append(y)

        x = self.k(x)
        y = self.flat(x)
        temp.append(y)

        x = self.l(x)
        y = self.flat(x)
        temp.append(y)

        x = self.m(x)
        y = self.flat(x)
        temp.append(y)

        x = self.n(x)
        y = self.flat(x)
        temp.append(y)

        x = self.o(x)
        y = self.flat(x)
        temp.append(y)

        x = self.p(x)
        y = self.flat(x)
        temp.append(y)

        x = self.q(x)
        y = self.flat(x)
        temp.append(y)

        x = self.r(x)
        y = self.flat(x)
        temp.append(y)

        x = self.s(x)

        features = torch.stack(temp, dim=0).permute(1,0,2)
        
        return features

    def flat(self,x):
        batch = x.shape[0]
        n_channel = x.shape[1]
        dim = x.shape[2]
        pool = nn.AvgPool2d((dim, dim))
        # pool = nn.MaxPool2d((dim, dim))
        x = pool(x)
        x = x.view(batch, -1)
        n = 2048 - n_channel
        pad = torch.zeros((batch, n))
        if torch.cuda.is_available():
            pad = pad.cuda()
        x = torch.cat((x, pad), dim=1)

        return x


# class based on poly-paper
class LayerAttentionRes(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(LayerAttentionRes, self).__init__()
        self.layers = LayersModelRes(img_dim, embed_size)
        self.attention = EncoderImageAttentionRes(img_dim, embed_size, n_attention, no_imgnorm)

    def forward(self, images):
        layer_features = self.layers.forward(images)
        features = self.attention(layer_features)

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

        super(LayerAttentionRes, self).load_state_dict(new_state)




class EncoderImageAttentionRes(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(EncoderImageAttentionRes, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.w1 = nn.Linear(img_dim, int(img_dim/2),bias=False)
        self.w2 = nn.Linear(int(img_dim/2), n_attention,  bias=False)
        self.w3 = nn.Linear(img_dim, embed_size, bias=True)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)


    def forward(self, images):
        """Extract image feature vectors."""
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

        super(EncoderImageAttentionRes, self).load_state_dict(new_state)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
