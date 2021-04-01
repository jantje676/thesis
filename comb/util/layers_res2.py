it import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict

""" Resnest layers for layers-attention-SCAN"""
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
        self.t = nn.Linear(2048, 2048)
    def forward1(self, x):
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

    # for glob testing, can be removed if doesnt work
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
        x = self.t(x)
        features = torch.stack(temp, dim=0).permute(1,0,2)

        return features, x

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
