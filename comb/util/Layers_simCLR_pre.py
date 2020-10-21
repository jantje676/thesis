import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append('../../')
sys.path.append('/home/kgoei/thesis')

from SimCLR_pre.modules.simclr import SimCLR


class Layers_simCLR_pre(nn.Module):
    def __init__(self, args, device, feature_dim=4096):
        super(Layers_simCLR_pre, self).__init__()
        net = SimCLR(args)
        print(net)
        exit()
        net.load_state_dict(torch.load(args.checkpoint_simCLR_pre, map_location=torch.device(device)))

        self.feature_dim = feature_dim
        self.a = nn.Sequential(net.encoder.conv1, net.encoder.bn1, net.encoder.relu, net.encoder.maxpool)
        self.b = net.encoder.layer1 #f1
        self.c = net.encoder.layer2 #f2
        self.d = net.encoder.layer3[0]
        self.e = net.encoder.layer3[1] #f3
        self.f = net.encoder.layer3[2]
        self.g = net.encoder.layer3[3] #f4
        self.h = net.encoder.layer3[4]
        self.i = net.encoder.layer3[5] #f5
        self.j = net.encoder.layer4[0] #f6
        self.k = net.encoder.layer4[1]
        self.l = net.encoder.layer4[2] #f7
        self.m = net.encoder.avgpool

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
