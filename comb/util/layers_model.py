import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np

class LayersModel(nn.Module):
    def __init__(self):
        super(LayersModel, self).__init__()
        self.net = models.alexnet(pretrained=True)

        self.relu = nn.ReLU(inplace=True)
        self.a = self.net.features[0]
        self.b = self.net.features[2]
        self.c = self.net.features[3]
        self.d = self.net.features[5]
        self.e = self.net.features[6] #
        self.f = self.net.features[8] #
        self.g = self.net.features[10] #
        self.h = self.net.features[12] #
        self.i = self.net.avgpool #
        self.j = self.net.classifier[0]
        self.k = self.net.classifier[1] #
        self.l = self.net.classifier[3]
        self.m = self.net.classifier[4] #


    def forward(self, x):
        temp = []
        x = self.a(x)
        x = self.relu(x)
        x = self.b(x)
        x = self.c(x)

        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        x = self.d(x)
        x = self.e(x) #

        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        x = self.f(x) #

        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        x = self.g(x) #

        y = flat(x)
        temp.append(y)

        x = self.relu(x)
        x = self.h(x) #

        y = flat(x)
        temp.append(y)

        x = self.i(x)
        x = self.j(x)
        x = x.view(1 ,-1)
        x = self.k(x) #

        y = x.squeeze(0).detach().cpu().numpy()
        temp.append(y)

        x = self.relu(x)
        x = self.l(x)
        x = self.m(x) #

        y = x.squeeze(0).detach().cpu().numpy()
        temp.append(y)

        features = np.stack(temp, axis=0)
        return features
def flat(x):
    dim = x.shape[2]
    pool = nn.AvgPool2d((dim, dim))
    x = pool(x)
    x = x.view(-1)
    x = x.detach().cpu().numpy()

    n = 4096 - x.shape[0]
    x = np.pad(x, (0, n), 'constant')
    return x
