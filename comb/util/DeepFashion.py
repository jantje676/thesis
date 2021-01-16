
from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import AttrPredictor, CatePredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor

import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append('/home/kgoei/thesis/comb/util')
sys.path.append('../')
# class based on deepfashion
class LayersAttr(nn.Module):
    def __init__(self, checkpoint):
        super(LayersAttr, self).__init__()
        self.feature_dim = 2048
        # config = '{}/DeepFashion/global_predictor_resnet.py'.format(checkpoint)
        # checkpoint = '{}/DeepFashion/checkpoint/epoch_40.pth'.format(checkpoint)

        config = 'DeepFashion/global_predictor_resnet.py'
        checkpoint =  'DeepFashion/checkpoint/epoch_40.pth'

        net = basis_model(config, checkpoint)

        self.a = nn.Sequential(net.backbone.conv1, net.backbone.bn1, net.backbone.relu, net.backbone.maxpool)
        self.b = net.backbone.layer1 #f1
        self.c = net.backbone.layer2 #f2
        self.d = net.backbone.layer3[0]
        self.e = net.backbone.layer3[1] #f3
        self.f = net.backbone.layer3[2]
        self.g = net.backbone.layer3[3] #f4
        self.h = net.backbone.layer3[4]
        self.i = net.backbone.layer3[5] #f5
        self.j = net.backbone.layer4[0] #f6
        self.k = net.backbone.layer4[1]
        self.l = net.backbone.layer4[2] #f7


        # self.a = net.backbone.conv1
        # self.b = nn.Sequential(net.backbone.bn1, net.backbone.relu, net.backbone.maxpool)
        # self.c = net.backbone.layer1[0]
        # self.d = net.backbone.layer1[1]
        # self.e = net.backbone.layer1[2]
        # self.f = net.backbone.layer2[0]
        # self.g = net.backbone.layer2[1]
        # self.h = net.backbone.layer2[2]
        # self.i = net.backbone.layer2[3]
        # self.j = net.backbone.layer3[0]
        # self.k = net.backbone.layer3[1]
        # self.l = net.backbone.layer3[2]
        # self.m = net.backbone.layer3[3]
        # self.n = net.backbone.layer3[4]
        # self.o = net.backbone.layer3[5]
        # self.p = net.backbone.layer4[0]
        # self.q = net.backbone.layer4[1]
        # self.r = net.backbone.layer4[2]
        # self.s = net.global_pool.avgpool
        # self.t = net.global_pool.global_layers[0]
        # self.u = net.global_pool.global_layers[1]
        # self.v = net.global_pool.global_layers[2]
        # self.w = net.global_pool.global_layers[3]
        # self.x = net.global_pool.global_layers[4]

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

def basis_model(config, checkpoint):
    cfg = Config.fromfile(config)

    model = build_predictor(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model
