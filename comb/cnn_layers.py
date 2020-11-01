import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import math
import torchvision.transforms as transforms

"""
Spatial transformer module that can be used in the SCAN model to learn automatic region extraction
"""

class CNN_layers(nn.Module):
    def __init__(self, n_detectors, embed_size, pretrained_alex, net):
        super(CNN_layers, self).__init__()
        self.n_detectors = n_detectors
        # conv nets
        self.conv = nn.ModuleList(retrieve_convnets(self.n_detectors, embed_size, pretrained_alex, net=net))

    def forward(self, x):
        # transform the input
        batch_size = x.shape[0]

        x = x.repeat(self.n_detectors, 1, 1, 1)

        stack = []
        for i in range(self.n_detectors):
            conv = self.conv[i]
            part_x = conv(x[i * batch_size : (i + 1) * batch_size])
            stack.append(part_x)
        temp = torch.stack(stack, 1)
        return temp

def retrieve_convnets(n_detectors, embed_size, pretrained_alex, net="alex"):
    conv = []
    for i in range(n_detectors):
        if net == "alex":

            temp_alex = models.alexnet(pretrained=pretrained_alex)

            temp_alex.classifier = nn.Sequential(*[temp_alex.classifier[i] for i in range(5)],nn.ReLU(), nn.Linear(4096, embed_size))
            if torch.cuda.is_available():
                temp_alex = temp_alex.cuda()
            conv.append(temp_alex)

    return conv

def check_image(x, indx, n_detectors):

    f, axarr = plt.subplots(n_detectors,1)
    batch = int(x.shape[0]/n_detectors)
    images = []

    print(x.shape)
    # find images
    for i in range(n_detectors):
        images.append(x[indx + i * batch].data.numpy().transpose(1,2,0))

    # plot images
    for i in range(n_detectors):
        axarr[i].imshow(images[i])

    plt.show()
