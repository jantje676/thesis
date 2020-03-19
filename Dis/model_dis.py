import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda
from model import BetaVAE_H, BetaVAE_B
from math import log, exp, sqrt


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kl_divergence(mu1, mu2, logvar1, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    kl = torch.log(torch.sqrt(var2)/torch.sqrt(var1)) + ((var1 + (mu1 - mu2)**2 )/ (2 * var2)) - 0.5
    return kl

def find_indexes(kl):
    min = torch.min(kl).item()
    max = torch.max(kl).item()

    threshold = (min + max) / 2
    indexes = (kl < threshold)
    indexes = indexes[0].nonzero()
    return indexes

class Dis_Net(nn.Module):
    def __init__(self, z_dim=10, nc=3, image_width=64, image_height=64):
        super(Dis_Net, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.image_width = image_width
        self.image_height = image_height
        self.net1 = BetaVAE_H(self.z_dim, self.nc, self.image_width, self.image_height)
        self.net2 = BetaVAE_H(self.z_dim, self.nc, self.image_width, self.image_height)


    def forward(self, x1, x2):
        distributions1 = self.net1._encode(x1)
        mu1 = distributions1[:, :self.z_dim]
        logvar1 = distributions1[:, self.z_dim:]

        distributions2 = self.net2._encode(x2)
        mu2 = distributions2[:, :self.z_dim]
        logvar2 = distributions2[:, self.z_dim:]

        kl = kl_divergence(mu1, mu2, logvar1, logvar2)

        index = find_indexes(kl)

        total_mu1 = mu1[0][index]
        total_mu2 = mu2[0][index]
        avererage_mu = (mu1[0][index] + mu2[0][index]) / 2

        mu1[0][index] = avererage_mu
        mu2[0][index] = avererage_mu
        average_logvar = (logvar1[0][index] + logvar2[0][index] )/ 2

        logvar1[0][index] = average_logvar
        logvar2[0][index] = average_logvar

        z1 = reparametrize(mu1, logvar1)
        z2 = reparametrize(mu2, logvar2)

        x_recon1 = self.net1._decode(z1)
        x_recon2 = self.net2._decode(z2)

        return (x_recon1, x_recon2), (mu1, mu2 ), (logvar1, logvar2)
