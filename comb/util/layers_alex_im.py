import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
import math

class LayersModel4(nn.Module):
    def __init__(self, img_dim=4096, embed_size=1024, trained_dresses=False, checkpoint_path=None):
        super(LayersModel4, self).__init__()
        net = models.alexnet(pretrained=True)
        if trained_dresses:
            print("Loading pretrained model on dresses")
            checkpoint = torch.load(checkpoint_path)
            weights = checkpoint["model"]
            del weights['classifier.6.weight']
            del weights['classifier.6.bias']
            net.load_state_dict(checkpoint["model"], strict=False)

        self.relu = nn.ReLU(inplace=False)
        self.a = net.features[0] #
        self.b = net.features[2]
        self.c = net.features[3] #
        self.d = net.features[5]
        self.e = net.features[6] #
        self.f = net.features[8] #
        self.g = net.features[10] #
        self.h = net.features[12] #
        self.i = net.avgpool #
        self.j = net.classifier[0]
        self.k = net.classifier[1] #
        self.l = net.classifier[3]
        self.m = net.classifier[4] #
        self.fc = nn.Linear(img_dim, img_dim)

        self.init_weights()

    def forward1(self, x):

        batch = x.shape[0]
        temp = []
        x = self.a(x)
        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        y = flat(x)
        temp.append(y)

        x = self.b(x)
        y = flat(x)
        temp.append(y)

        x = self.c(x)
        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        y = flat(x)
        temp.append(y)

        x = self.d(x)
        y = flat(x)
        temp.append(y)

        x = self.e(x) #
        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        y = flat(x)
        temp.append(y)

        x = self.f(x) #
        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        y = flat(x)
        temp.append(y)

        x = self.g(x) #
        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        y = flat(x)
        temp.append(y)

        x = self.h(x) #
        y = flat(x)
        temp.append(y)

        x = self.i(x)
        y = flat(x)
        temp.append(y)

        x = self.j(x)
        y = flat(x)
        temp.append(y)

        x = x.view(batch ,-1)
        x = self.k(x) #
        temp.append(x)

        x = self.relu(x)
        temp.append(y)

        x = self.l(x)
        temp.append(y)

        x = self.m(x) #
        temp.append(x)

        features = torch.stack(temp, dim=0).permute(1,0,2)

        return features

    def forward(self, x):
        features = self.forward1(x)
        features = self.fc(features)
        return features

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(LayersModel4, self).load_state_dict(new_state)

def flat(x):
    batch = x.shape[0]
    n_channel = x.shape[1]
    dim = x.shape[2]
    pool = nn.AvgPool2d((dim, dim))
    # pool = nn.MaxPool2d((dim, dim))
    x = pool(x)
    x = x.view(batch, -1)
    n = 4096 - n_channel
    pad = torch.zeros((batch, n))
    if torch.cuda.is_available():
        pad = pad.cuda()
    x = torch.cat((x, pad), dim=1)


    return x


# class based on poly-paper
class LayerAttention4(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(LayerAttention4, self).__init__()
        self.layers = LayersModel4(img_dim, embed_size)
        self.attention = EncoderImageAttention4(img_dim, embed_size, n_attention, no_imgnorm)

        # checkpoint_path = "../train_models/runAlex/train1/checkpoint/model_best.pth.tar"
        checkpoint_path = "/home/kgoei/runAlex/test1/checkpoint/model_best.pth.tar"

        print("Loading pretrained model on dresses")
        net = models.alexnet(pretrained=True)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        weights = checkpoint["model"]
        del weights['classifier.6.weight']
        del weights['classifier.6.bias']
        net.load_state_dict(checkpoint["model"], strict=False)
        net.classifier = nn.Sequential(*[net.classifier[i] for i in range(5)])
        self.glob_net = net

        for param in self.glob_net.parameters():
            param.requires_grad = False

    def forward(self, images):
        layer_features = self.layers.forward1(images)
        glob_features = self.glob_net(images)

        features = self.attention(layer_features, glob_features)

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

        super(LayerAttention4, self).load_state_dict(new_state)




class EncoderImageAttention4(nn.Module):

    def __init__(self, img_dim, embed_size, n_attention, no_imgnorm=False):
        super(EncoderImageAttention4, self).__init__()
        self.embed_size = embed_size
        self.img_dim = img_dim
        self.no_imgnorm = no_imgnorm
        self.n_attention = n_attention
        self.w1 = nn.Linear(img_dim, int(img_dim/2),bias=False)
        self.w2 = nn.Linear(img_dim, n_attention,  bias=False)
        self.w3 = nn.Linear(img_dim, embed_size, bias=True)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)


    def forward(self, images, glob_features):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        keys = self.w1(images)
        glob_features = glob_features.unsqueeze(dim=1).expand(-1,int(self.img_dim/2), -1 )
        queries = self.w2(glob_features)

        temp_attn = torch.bmm(queries.transpose(1,2), keys.transpose(1,2))
        temp_attn = temp_attn / math.sqrt(self.img_dim)
        attention = F.softmax(temp_attn, dim=2)
        feat = torch.bmm(attention, images)

        feat = l2norm(feat, dim=-1)

        feat = self.w3(feat)
        feat = torch.sigmoid(feat)

        return feat

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageAttention4, self).load_state_dict(new_state)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
