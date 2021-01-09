import torch


"""
Diferent loss functions to measure the diversity between features in the SCAN model
"""

def cosine_loss(theta, im):
    num = torch.bmm(im, im.permute(0,2,1))
    norm = torch.norm(im, dim=2).unsqueeze(dim=2)
    denom = torch.bmm(norm, norm.permute(0,2,1))
    sim_im = (num / (denom).clamp(min=1e-08))
    sim_im = 1 + sim_im
    loss_div = torch.triu(sim_im, diagonal=1)
    loss_div = loss_div.sum() * theta
    return loss_div

# sigma should be >10000
def euclidean_heat_loss(theta, im, sigma, n_detectors):
    im = im.unsqueeze(dim=2)
    im_exp = im.repeat(1,1,n_detectors,1)
    im_trans = torch.transpose(im_exp, 1,2)
    diff = im_exp - im_trans
    norm = torch.norm(diff, dim=3)
    pow = ((norm **2) / sigma) * -1
    sim_im = torch.exp(pow)
    loss_div = torch.triu(sim_im, diagonal=1)
    loss_div = loss_div.sum() * theta
    return loss_div

def euclidean_loss(theta, im, n_detectors):
    im = im.unsqueeze(dim=2)
    im_exp = im.repeat(1,1,n_detectors,1)
    im_trans = torch.transpose(im_exp, 1,2)
    diff = im_exp - im_trans
    norm = torch.norm(diff, dim=3)
    sim_im = 1/(norm **2)
    loss_div = torch.triu(sim_im, diagonal=1)
    loss_div = loss_div.sum() * theta
    return loss_div

# theta should be low: 0.0000001
def ssd(theta, im):
    kernel = torch.bmm(im, torch.transpose(im, 1,2 ))
    eigen = torch.symeig(kernel, eigenvectors=True)
    eigen_values = eigen[0]
    sim_im = (eigen_values - 1) ** 2
    loss_div = sim_im.sum() * theta
    return loss_div

# features should be normalized otherwise they dont work
def dpp(theta, im):

    kernel = torch.bmm(im, torch.transpose(im, 1,2 ))
    det = kernel.det()
    log_det = 1/torch.log(det)
    loss_div = log_det.sum() * theta
    return loss_div

def weight_loss(net, diversity_loss, theta, sigma):
    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []
    w6 = []
    w7 = []
    w8 = []


    for cnn in net.conv:
        w1.append(torch.flatten(cnn.features[0].weight))
        w2.append(torch.flatten(cnn.features[3].weight))
        w3.append(torch.flatten(cnn.features[6].weight))
        w4.append(torch.flatten(cnn.features[8].weight))
        w5.append(torch.flatten(cnn.features[10].weight))
        w6.append(torch.flatten(cnn.classifier[1].weight))
        w7.append(torch.flatten(cnn.classifier[4].weight))
        w8.append(torch.flatten(cnn.classifier[6].weight))

    weights = [w1, w2, w3, w4, w5, w6, w7, w8]
    loss = 0
    for w in weights:
        im = torch.stack(w, dim=0).unsqueeze(dim=0)
        if diversity_loss == "cos":
            loss += cosine_loss(theta, im)
        elif diversity_loss == "euc_heat":
            loss += euclidean_heat_loss(theta, im, sigma, 7)
        elif diversity_loss == "euc":
            loss += euclidean_loss(theta, im, 7)
        elif self.opt.diversity_loss == "ssd":
            loss += ssd(theta, im)
        elif self.opt.diversity_loss == "dpp":
            loss += dpp(theta, im)

    return loss
