
from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append('/home/kgoei/thesis/comb/util')
import PIL

# class based on deepfashion
class LayersModelAttr(nn.Module):
    def __init__(self):
        super(LayersModelAttr, self).__init__()
        config = '/home/kgoei/thesis/comb/util/DeepFashion/global_predictor_resnet.py'
        checkpoint = '/home/kgoei/thesis/comb/util/DeepFashion/checkpoint/epoch_40.pth'

        # config = 'util/DeepFashion/global_predictor_resnet.py'
        # checkpoint = 'util/DeepFashion/checkpoint/epoch_40.pth'

        net = basis_model(config, checkpoint)
        self.a = net.backbone.conv1
        self.b = nn.Sequential(net.backbone.bn1, net.backbone.relu, net.backbone.maxpool)
        self.c = net.backbone.layer1[0]
        self.d = net.backbone.layer1[1]
        self.e = net.backbone.layer1[2]
        self.f = net.backbone.layer2[0]
        self.g = net.backbone.layer2[1]
        self.h = net.backbone.layer2[2]
        self.i = net.backbone.layer2[3]
        self.j = net.backbone.layer3[0]
        self.k = net.backbone.layer3[1]
        self.l = net.backbone.layer3[2]
        self.m = net.backbone.layer3[3]
        self.n = net.backbone.layer3[4]
        self.o = net.backbone.layer3[5]
        self.p = net.backbone.layer4[0]
        self.q = net.backbone.layer4[1]
        self.r = net.backbone.layer4[2]
        self.s = net.global_pool.avgpool
        self.t = net.global_pool.global_layers[0]
        self.u = net.global_pool.global_layers[1]
        self.v = net.global_pool.global_layers[2]
        self.w = net.global_pool.global_layers[3]
        self.x = net.global_pool.global_layers[4]

    def forward1(self, x):
        temp = []
        x = transforms.Resize((224, 224),interpolation=PIL.Image.NEAREST)(x)
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
        x = x.view(x.size(0), -1)
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
        features = torch.stack(temp, dim=0).permute(1,0,2)
        return features

    def flat(self,x):
        batch = x.shape[0]
        n_channel = x.shape[1]
        if len(x.shape) > 2:
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


def basis_model(config, checkpoint):
    cfg = Config.fromfile(config)
    model = build_predictor(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model
