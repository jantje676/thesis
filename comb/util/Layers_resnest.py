import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict


class Layers_resnest(nn.Module):
    def __init__(self, feature_dim=4096):
        super(Layers_resnest, self).__init__()
        self.feature_dim = feature_dim
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
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

    def forward(self, x):
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
        x = self.m(x)
        features = torch.stack(temp, dim=0).permute(1,0,2)
        return features

    def flat(self, x):
        batch = x.shape[0]
        n_channel = x.shape[1]
        dim = x.shape[2]
        pool = nn.AvgPool2d((dim, dim))
        x = pool(x)

        x = x.view(batch, -1)
        n = self.feature_dim - n_channel
        pad = torch.zeros((batch, n))
        if torch.cuda.is_available():
            pad = pad.cuda()
        x = torch.cat((x, pad), dim=1)
        return x
