import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict

""" Deep Res for Layers-Attention-SCAN"""
class LayersModelResDeep(nn.Module):
    def __init__(self, feature_dim=2048, trained_dresses=False, checkpoint_path=None):
        super(LayersModelResDeep, self).__init__()
        self.feature_dim = feature_dim
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
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
        self.p = net.layer3[6]
        self.q = net.layer3[7]
        self.r = net.layer3[8]
        self.s = net.layer3[9]
        self.t = net.layer3[10]
        self.u = net.layer3[11]
        self.v = net.layer3[12]
        self.w = net.layer3[13]
        self.x = net.layer3[14]
        self.y = net.layer3[15]
        self.z = net.layer3[16]
        self.aa = net.layer3[17]
        self.ab = net.layer3[18]
        self.ac = net.layer3[19]
        self.ad = net.layer3[20]
        self.ae = net.layer3[21]
        self.af = net.layer3[22]
        self.ag = net.layer4[0]
        self.ah = net.layer4[1]
        self.ai = net.layer4[2]
        self.aj = net.avgpool

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
        y = self.flat(x)
        temp.append(y)

        x = self.t(x)
        y = self.flat(x)
        temp.append(y)

        x = self.u(x)
        y = self.flat(x)
        temp.append(y)

        x = self.v(x)
        y = self.flat(x)
        temp.append(y)

        x = self.w(x)
        y = self.flat(x)
        temp.append(y)

        x = self.x(x)
        y = self.flat(x)
        temp.append(y)

        x = self.y(x)
        y = self.flat(x)
        temp.append(y)

        x = self.z(x)
        y = self.flat(x)
        temp.append(y)

        x = self.aa(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ab(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ac(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ad(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ae(x)
        y = self.flat(x)
        temp.append(y)

        x = self.af(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ag(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ah(x)
        y = self.flat(x)
        temp.append(y)

        x = self.ai(x)
        y = self.flat(x)
        temp.append(y)

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
