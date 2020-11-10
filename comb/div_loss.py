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

def euclidean_loss(theta, im, sigma):
    im = im.unsqueeze(dim=2)
    im_exp = im.repeat(1,1,7,1)
    im_trans = torch.transpose(im_exp, 1,2)
    diff = im_exp - im_trans

    norm = torch.norm(diff, dim=3)

    pow = ((norm **2) / sigma) * -1

    sim_im = torch.exp(pow)

    loss_div = torch.triu(sim_im, diagonal=1)
    loss_div = loss_div.sum() * theta

    return loss_div
