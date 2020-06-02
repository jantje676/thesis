import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import math



class STN(nn.Module):
    def __init__(self, n_detectors, embed_size, pretrained_alex, rectangle):
        super(STN, self).__init__()
        self.n_detectors = n_detectors
        start_transformation = init_trans(n_detectors, rectangle)

        # conv nets
        self.conv = nn.ModuleList(retrieve_convnets(self.n_detectors, embed_size, pretrained_alex))

        # Spatial transformer localization-network
        self.localization = models.alexnet()
        self.localization = self.localization.features

        # create n transformation parameters according to n_detectors
        if rectangle:
            self.bottle = 26880
        else:
            self.bottle = 12544

        self.fc_loc = nn.Sequential(
            nn.Linear(self.bottle, 6125),
            nn.Linear(6125, 32),
            nn.Linear(32, 6 * self.n_detectors)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(start_transformation)

        self.mask = get_mask(n_detectors)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.bottle)
        theta = self.fc_loc(xs)
        theta = theta * self.mask
        theta = theta.view(-1, 2, 3)
        indices = get_indices(self.n_detectors, x.shape[0])
        theta = torch.index_select(theta, 0, indices)

        x = x.repeat(self.n_detectors, 1,1, 1)

        grid = F.affine_grid(theta, (x.shape[0], 3, 256, 256))
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
        return temp


def get_indices(n_detectors, batch_size):
    # get the indices so that I can resort theta using index select
    temp = []
    for i in range(n_detectors):
        ind = torch.arange(i,batch_size*n_detectors, n_detectors)
        temp.append(ind)
    indices = torch.cat(temp)
    if torch.cuda.is_available():
        indices = indices.cuda()
    return indices

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

def init_trans(n_detectors, rectangle):
    # for now take 2 columns and make rows flexible according to number of detectors
    column = 2
    row = math.floor((n_detectors - 1)/column)
    step_column = 1/(column - 1)
    step_row = 1/ (row - 1)

    # for now this zoom is good
    if rectangle:
        z_x = 1
    else:
        z_x = 0.5

    z_y = 0.5

    s_x = [1]
    s_y = [1]
    t_x = [0]
    t_y = [0]
    # add identity for odd number, if even skip this!
    if n_detectors % 2 == 0:
        s_x.append(0.7)
        s_y.append(0.7)
        t_x.append(0)
        t_y.append(0)


    # init in tiles
    for i in range(column):
        for j in range(row):
            t_x.append(-0.5 + step_column*i)
            t_y.append(-0.5 + step_row*j)
            s_x.append(z_x)
            s_y.append(z_y)


    temp = []
    for i in range(n_detectors):
        t = torch.tensor([s_x[i], 0, t_x[i], 0, s_y[i] ,t_y[i]], dtype=torch.float)
        temp.append(t)

    trans = torch.cat(temp)

    return trans

def get_mask(n_detectors):
    start_mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float)
    mask = start_mask.repeat(n_detectors)
    return mask

def check_image(x, indx, n_detectors):

    f, axarr = plt.subplots(n_detectors,1)
    batch = int(x.shape[0]/n_detectors)
    images = []

    # find images
    for i in range(n_detectors):
        images.append(x[indx + i * batch].data.numpy().transpose(1,2,0))

    # plot images
    for i in range(n_detectors):
        axarr[i].imshow(images[i],  interpolation='nearest')

    plt.show()
