import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.init
import torch.nn.functional as F
from collections import OrderedDict

"""Deep res-101 for Layers-SCAN and create of layer features"""

class LayersScanResDeep(nn.Module):
    def __init__(self, feature_dim=2048, trained_dresses=False, checkpoint_path=None):
        super(LayersScanResDeep, self).__init__()
        self.feature_dim = feature_dim
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)

        self.a = net.conv1
        self.b = nn.Sequential(net.bn1, net.relu, net.maxpool)
        self.c = net.layer1[0]
        self.d = net.layer1[1]#1
        self.e = net.layer1[2]
        self.f = net.layer2[0]
        self.g = net.layer2[1]
        self.h = net.layer2[2]
        self.i = net.layer2[3]
        self.j = net.layer3[0]#2
        self.k = net.layer3[1]
        self.l = net.layer3[2]
        self.m = net.layer3[3]
        self.n = net.layer3[4]
        self.o = net.layer3[5]
        self.p = net.layer3[6] #3
        self.q = net.layer3[7]
        self.r = net.layer3[8]
        self.s = net.layer3[9]
        self.t = net.layer3[10]
        self.u = net.layer3[11]#4
        self.v = net.layer3[12]
        self.w = net.layer3[13]
        self.x = net.layer3[14]
        self.y = net.layer3[15]
        self.z = net.layer3[16]#5
        self.aa = net.layer3[17]
        self.ab = net.layer3[18]
        self.ac = net.layer3[19]
        self.ad = net.layer3[20]
        self.ae = net.layer3[21] #6
        self.af = net.layer3[22]
        self.ag = net.layer4[0]
        self.ah = net.layer4[1]
        self.ai = net.layer4[2] #7
        self.aj = net.avgpool
        self.fc = nn.Linear(feature_dim, 1024)
        self.init_weights()


    def forward1(self, x):
        temp = []

        x = self.a(x)
        x = self.b(x)

        x = self.c(x)

        x = self.d(x)
        y = self.flat(x)
        temp.append(y)

        x = self.e(x)
        x = self.f(x)
        x = self.g(x)
        x = self.h(x)
        x = self.i(x)

        x = self.j(x)
        y = self.flat(x)
        temp.append(y)

        x = self.k(x)
        x = self.l(x)
        x = self.m(x)
        x = self.n(x)
        x = self.o(x)

        x = self.p(x)
        y = self.flat(x)
        temp.append(y)

        x = self.q(x)
        x = self.r(x)
        x = self.s(x)
        x = self.t(x)
        x = self.u(x)

        x = self.v(x)
        y = self.flat(x)
        temp.append(y)

        x = self.w(x)
        x = self.x(x)
        x = self.y(x)

        x = self.z(x)
        y = self.flat(x)
        temp.append(y)

        x = self.aa(x)
        x = self.ab(x)
        x = self.ac(x)
        x = self.ad(x)

        x = self.ae(x)
        y = self.flat(x)
        temp.append(y)

        x = self.af(x)
        x = self.ag(x)
        x = self.ah(x)

        x = self.ai(x)
        y = self.flat(x)
        temp.append(y)

        features = torch.stack(temp, dim=0).permute(1,0,2)

        return features

    def forward(self, x):
        features = self.forward1(x)
        features = self.fc(features)
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

        super(LayersScanResDeep, self).load_state_dict(new_state)
