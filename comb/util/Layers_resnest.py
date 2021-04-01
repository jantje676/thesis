import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict


"""ResNeSt-50 layers model for Layers-SCAN"""
class Layers_resnest(nn.Module):
    def __init__(self, img_dim=2048, embed_size=1024, trained_dresses=False, checkpoint_path=None):
        super(Layers_resnest, self).__init__()
        self.img_dim = img_dim
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        if trained_dresses:
            checkpoint = torch.load(checkpoint_path)
            weights = checkpoint["model"]
            del weights['fc.weight']
            del weights['fc.bias']
            net.load_state_dict(checkpoint["model"], strict=False)

        self.a = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.b = net.layer1 #f1
        self.c = net.layer2 #f2
        self.d = net.layer3[0]
        self.e = net.layer3[1] #f3
        self.f = net.layer3[2]
        self.g = net.layer3[3] #f4
        self.h = net.layer3[4]
        self.i = net.layer3[5] #f5
        self.j = net.layer4[0] #f6
        self.k = net.layer4[1]
        self.l = net.layer4[2] #f7
        self.m = net.avgpool
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def forward1(self, x):
        temp = []
        x = self.a(x)
        x = self.b(x) #f1
        y = self.flat(x)
        temp.append(y)

        x = self.c(x)
        y = self.flat(x)
        temp.append(y)
        x = self.d(x)
        x = self.e(x)
        y = self.flat(x)
        temp.append(y)
        x = self.f(x)
        x = self.g(x)
        y = self.flat(x)
        temp.append(y)

        x = self.h(x)
        x = self.i(x)
        y = self.flat(x)
        temp.append(y)

        x = self.j(x)
        y = self.flat(x)
        temp.append(y)
        x = self.k(x)
        x = self.l(x)
        y = self.flat(x)
        temp.append(y)

        features = torch.stack(temp, dim=0).permute(1,0,2)
        return features

    def forward(self, x):
        features = self.forward1(x)
        features = self.fc(features)
        return features

    def flat(self, x):
        batch = x.shape[0]
        n_channel = x.shape[1]
        dim = x.shape[2]
        pool = nn.AvgPool2d((dim, dim))
        x = pool(x)

        x = x.view(batch, -1)
        n = self.img_dim - n_channel
        pad = torch.zeros((batch, n))
        if torch.cuda.is_available():
            pad = pad.cuda()
        x = torch.cat((x, pad), dim=1)
        return x

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

        super(Layers_resnest, self).load_state_dict(new_state)
