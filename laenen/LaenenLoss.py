import torch.nn as nn
import torch
from sklearn import preprocessing
import numpy as np

class LaenenLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin, n, switch, beta, gamma):
        super(LaenenLoss, self).__init__()
        self.relu = nn.ReLU()
        self.beta = beta
        self.gamma = gamma

        # hyperparameters
        self.switch = switch
        self.n = n
        self.margin = margin

    def forward(self, epoch, img_emb, cap_emb, cap_l, kmeans_features, kmeans_emb, cluster_loss):
        n_frag = img_emb.size(1)
        batch_size = img_emb.size(0)
        n_caption = cap_emb.size(0)

        sims = torch.einsum('bik,ljk->blij', img_emb, cap_emb)

        c_frag_loss = self.c_frag(sims, cap_l, epoch, n_frag, batch_size, n_caption)
        c_glob_loss = self.c_glob(sims, cap_l, n_frag, batch_size, n_caption)
        loss = self.beta * c_glob_loss + c_frag_loss

        if cluster_loss:
            c_cluster = self.c_cluster(kmeans_features, kmeans_emb, sims, img_emb, cap_emb, cap_l, features)
            loss += self.gamma * c_cluster

        return loss


    def c_glob(self, sims, cap_l, n_frag, batch_size, n_caption):
        sims, _ = torch.max(sims, dim=2)

        sims = sims.sum(dim=2)

        thres_image = get_thres(cap_l, self.n).unsqueeze(0).expand(batch_size, -1)

        sims = sims * thres_image
        diag = torch.diagonal(sims, 0)

        image_diag = diag.unsqueeze(1).expand(-1, n_caption)
        cap_diag = diag.unsqueeze(0).expand(batch_size, -1)

        score_image = sims - image_diag + self.margin
        score_cap = sims - cap_diag + self.margin

        score_image = self.relu(score_image)
        score_cap = self.relu(score_cap)

        return score_cap.sum()+ score_image.sum()

    def sim_val(self, img, cap, l):
        batch_size = img.size(0)
        sims = torch.einsum('bik,ljk->blij', img, cap)

        sims, _ = torch.max(sims, dim=2)

        sims = sims.sum(dim=2)

        thres_image = get_thres(l, self.n).unsqueeze(0).expand(batch_size, -1)

        sims = sims * thres_image

        return sims

    def c_frag(self, sims, cap_l, epoch, n_frag, batch_size, n_caption):
        loss = 0

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_l[i]

            sims_i = sims[:,i,:,:n_word]

            # first n epochs fix the constants y_ij
            if epoch < self.switch:
                y_i = init_y(sims_i, i)
            # after let the model optimize y_ij with the heuristic sign
            else:
                y_i = sign(sims_i, i)

            sims_i = 1 - (y_i * sims_i)
            score = torch.sum(self.relu(sims_i))
            loss += score
        return loss


    def sim_pair(self, img, cap, s_l):
        batch_size = cap.size(0)
        cap_l = cap.size(1)
        n_frag = img.size(1)

        # switch axes to use matmul on the 2nd and 3rd dimension
        cap = cap.permute(0,2,1)

        # calculate similarity between the two embeddings
        sims = torch.matmul(img , cap)

        sims, _ = torch.max(sims, dim=1)

        # sum over the dimensions
        sims = sims.sum(dim=1)

        # find threshold values
        thres = get_thres(s_l, self.n)

        # calculate sim_pair
        diag = sims* thres

        return diag

def get_thres(l, n):
    thres = (l + n)
    thres = float(1)/thres.to(dtype=torch.float)

    if torch.cuda.is_available():
        thres = thres.cuda()
    return thres


def sign(sims_i, i):
    y =  torch.ones(sims_i.shape, requires_grad=True) * -1
    temp_y = torch.sign(sims_i[i,:,:])

    n_frag = sims_i.size(1)
    n_word = sims_i.size(2)

    # check if at least one in every row has positive sign
    temp_sum = temp_y.sum(dim=0)

    sign_check = temp_sum > (n_frag * -1)

    if sign_check.sum() != n_word:
        for j in range(len(sign_check)):
            if sign_check[j] == False:
                i_max = torch.argmax(sims_i[i,:,j])
                temp_y[i_max, j] = 1


    y[i,:,:] = temp_y

    if torch.cuda.is_available():
        y = y.cuda()

    return y

# init y matrix with ones when image and word fragment are from the same pair
def init_y(sims_i, i):
    y =  torch.ones(sims_i.shape, requires_grad=True) * -1
    y[i, :, :] = 1

    if torch.cuda.is_available():
        y = y.cuda()
    return y


def c_cluster(self, kmeans_features, kmeans_emb, sims, img_emb, cap_emb, cap_l, features):
    batch_size = cap_emb.size(0)
    max_l = cap_emb.size(1)
    part1 = cluster1(features, kmeans_features)
    part2 = cluster2(img_emb, kmeans_emb, sims, cap_emb)

    part1 = part1.unsqueeze(1).expand(-1, batch_size, -1)
    part1 = part1.unsqueeze(3).expand(-1, -1, -1,max_l )

    loss = part1 * part2
    loss = torch.sum(loss)

    return loss

def cluster1(features, kmeans_features):
    dim = features.size(2)
    batch_size = features.size(0)
    n_frag = features.size(1)

    # reshape for sklearn
    im_norm = np.reshape(features, (-1, dim))

    im_norm = preprocessing.normalize(im_norm)

    # find labels of the clusters
    center_labels = kmeans_features.predict(im_norm)

    # retrieve nearest center vector
    centers = kmeans_features.cluster_centers_[center_labels]

    centers = np.reshape(centers, (batch_size, n_frag, dim ))
    im_norm = np.reshape(im_norm, (batch_size, n_frag, dim ))

    centers = torch.from_numpy(centers).float()
    im_norm = torch.from_numpy(im_norm).float()

    cos = cosine_similarity(im_norm, centers, dim=2)
    loss1 = 1 - cos
    return loss1


def cluster2(img_emb, kmeans_emb, sims, cap_emb):
    batch_size = img_emb.size(0)
    n_frag = img_emb.size(1)
    dim_emb = img_emb.size(2)

    # reshape for sklearn
    im_norm = np.reshape(img_emb.detach().numpy(), (-1, dim_emb))

    # bring to norm? > check if this is correct
    im_norm = preprocessing.normalize(im_norm)

    # find labels of the clusters
    center_labels = kmeans_emb.predict(im_norm)

    # retrieve nearest center vector
    centers = kmeans_emb.cluster_centers_[center_labels]

    centers = np.reshape(centers, (batch_size, n_frag, dim_emb ))
    centers = torch.from_numpy(centers).float()

    sims_center = torch.einsum('bik,ljk->blij', centers, cap_emb)

    loss2 = torch.abs(sims - sims_center)
    return loss2


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    temp = (w12 / (w1 * w2).clamp(min=eps))

    return temp.squeeze()
