import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import math
import torchvision.transforms as transforms
import kornia

"""
Architecture that trains n-cnns in parralel with diversity loss
"""

class CNN_layers(nn.Module):
    def __init__(self, n_detectors, embed_size, pretrained_alex, net, div_transform):
        super(CNN_layers, self).__init__()
        self.n_detectors = n_detectors
        # conv nets
        self.conv = nn.ModuleList(retrieve_convnets(self.n_detectors, embed_size, pretrained_alex, net=net))
        self.transforms = [to_hsv(),
                            kornia.filters.Sobel(),
                            remove_channel(0),
                            transforms.Grayscale(num_output_channels=3),
                            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4),
                            identity()]
        self.div_transform = div_transform

    def forward(self, x):
        # transform the input
        batch_size = x.shape[0]

        x = x.repeat(self.n_detectors, 1, 1, 1)

        stack = []
        for i in range(self.n_detectors):
            conv = self.conv[i]
            transform = self.transforms[i]
            temp = x[i * batch_size : (i + 1) * batch_size]

            if self.div_transform:
                temp = transform(temp)
            part_x = conv(temp)
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

class identity(object):
    def __call__(self,x):
        return  x

class to_hsv(object):
    def __call__(self,x):
        x = kornia.rgb_to_hsv(x)
        return  x

class remove_channel(object):
    def __init__(self, channel):
        self.channel = channel

    def __call__(self,x):
        x[:, self.channel, :, :] = torch.zeros(x.shape[0], x.shape[2], x.shape[3])
        return  x


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
