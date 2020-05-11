import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


class STN(nn.Module):
    def __init__(self, n_detectors):
        super(STN, self).__init__()
        self.n_detectors = n_detectors

        # conv nets
        self.conv = retrieve_convnets(self.n_detectors)

        # Spatial transformer localization-network
        self.localization = models.alexnet(pretrained=True)
        self.localization = self.localization.features


        # create n transformation parameters according to n_detectors
        self.fc_loc = nn.Sequential(
            nn.Linear(12544, 6125),
            nn.Linear(6125, 32),
            nn.Linear(32, 6 * self.n_detectors)
        )

        start_transformation = init_trans(n_detectors)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(start_transformation)

        self.mask = get_mask(n_detectors)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)

        xs = xs.view(-1, 12544)
        theta = self.fc_loc(xs)


        theta = theta * self.mask
        theta = theta.view(-1, 2, 3)
        x = x.repeat(self.n_detectors, 1,1, 1)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        batch_size = x.shape[0]
        x = self.stn(x)
        stack = []
        for i in range(self.n_detectors):
            conv = self.conv[i]
            part_x = conv(x[i * batch_size : (i + 1) * batch_size])
            stack.append(part_x)
        temp = torch.stack(stack, 1)
        print(temp.shape)
        return temp

def retrieve_convnets(n_detectors, net="alex"):
    conv = []
    for i in range(n_detectors):
        if net == "alex":
            temp_alex = models.alexnet(pretrained=True)
            temp_alex.classifier = nn.Sequential(*[temp_alex.classifier[i] for i in range(5)],nn.ReLU(), nn.Linear(4096, 1024))
            conv.append(temp_alex)

    return conv

def init_trans(n_detectors):
    identiy = torch.tensor([1, 0, 0, 0, 1 ,0], dtype=torch.float)
    trans = identiy.repeat(n_detectors)
    return trans

def get_mask(n_detectors):
    start_mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float)
    mask = start_mask.repeat(n_detectors)
    return mask

def check_image(img):
    img = img.data.numpy().transpose(1,2,0)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def shapen():
    a = torch.arange(1,4)
    b = torch.arange(1,4)
    print(a)
    print(b)
    # c = torch.cat([a,b],0)
    c = torch.stack([a,b], 1)
    print(c.shape)
    print(c)
    # c = c.view(2, 3).t()
    print(c)
    c = c.reshape(6)
    print(c)
