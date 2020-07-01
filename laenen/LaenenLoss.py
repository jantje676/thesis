import torch.nn as nn
import torch

class LaenenLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self):
        super(LaenenLoss, self).__init__()
        self.relu = nn.ReLU()


        # hyperparameters
        self.switch = 10
        self.n = 10
        self.margin = 0.2

    def forward(self, im, s, s_l, epoch, im_id, ids, image_diag, cap_diag):
        pair, pair_id = check_pair(im_id, ids)
        c_frag_loss = self.c_frag(im, s, s_l, epoch, pair, pair_id)
        c_glob_loss = self.c_glob(im, s, s_l, image_diag, cap_diag, pair, pair_id)
        loss =  c_glob_loss + c_frag_loss
        return loss


    def c_glob(self, im, s, s_l, image_diag, cap_diag, pair, pair_id):

        n_frag = im.size(1)
        n_caption = s.size(0)
        temp = []
        # reshape the images
        im = im.squeeze(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = s_l[i]
            cap_i = s[i, :n_word, :].contiguous()

            sim_cap = cap_i @ im.t()
            sim_cap = sim_cap.sum()
            temp.append(sim_cap)

        sims = torch.stack(temp)
        thres_image = get_thres(s_l, self.n, n_frag)
        thres_cap = get_thres(n_frag, self.n, s_l)
        sims_image = sims * thres_image
        sims_cap = sims * thres_cap
        score_image = sims_image - image_diag + self.margin
        score_cap = sims_cap - cap_diag + self.margin

        score_image = self.relu(score_image)
        score_cap = self.relu(score_cap)
        if pair:
            score_image[pair_id] = 0
            score_cap[pair_id] = 0

        return score_image.sum() + score_cap.sum()

    def c_frag(self, im, s, s_l, epoch, pair, pair_id):
        loss = 0
        n_frag = im.size(1)
        batch_size = im.size(0)
        n_caption = s.size(0)

        im = im.squeeze(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = s_l[i]
            cap_i = s[i, :n_word, :].contiguous()

            sim = cap_i @ im.t()

            # first n epochs fix the constants y_ij
            if epoch < self.switch:
                y_i = init_y(cap_i, im, i, n_frag, batch_size, pair, pair_id)
            # after let the model optimize y_ij with the heuristic sign
            else:
                y_i = sign(sim, i, n_frag, batch_size, n_word, pair, pair_id)

            sim = 1 - (y_i * sim)
            score = torch.sum(self.relu(sim))
            loss += score
        return loss


    def sim_pair(self, img_emb_pair, cap_emb_pair):

        # calculate the similairty score between the image and caption pair
        img_emb_pair = img_emb_pair.squeeze(0)
        cap_emb_pair = cap_emb_pair.squeeze(0)

        s_l = torch.tensor(cap_emb_pair.size(0))
        n_frag = img_emb_pair.size(0)

        sim_cap = cap_emb_pair @ img_emb_pair.t()
        sim_cap = sim_cap.sum()

        thres_image = get_thres(s_l, self.n, n_frag)
        thres_cap = get_thres(n_frag, self.n, s_l)

        # calculate sim_pair
        cap_diag = sim_cap * thres_cap
        image_diag = sim_cap * thres_image

        return cap_diag, image_diag

def get_thres(a, n, b):
    thres = (a + n) * b
    thres = float(1)/thres.to(dtype=torch.float)
    return thres

def sign(sim, i, n_frag, batch_size, n_word, pair, pair_id):
    y = torch.sign(sim)

    if pair and pair_id == i:
        # check if at least one in every row has positive sign
        temp_sum = y.sum(dim=1)
        sign_check = temp_sum > (n_frag * -1)
        if sign_check.sum() != n_word:
            for j in range(len(sign_check)):
                if sign_check[j] == False:
                    i_max = torch.argmax(sim[j])
                    y[j, i_max] = 1
    return y

# init y matrix with ones when image and word fragment are from the same pair
def init_y(cap_i, im, i, n_frag, batch, pair, pair_id):
    n_word = cap_i.size(0)

    if pair and pair_id == i:
        y =  torch.ones((n_word, n_frag * batch ), requires_grad=True)
    else:
        y =  torch.ones((n_word, n_frag * batch ), requires_grad=True) * -1
    return y

def check_pair(im_id, ids):
    pair = False
    pair_id = None

    if im_id in ids:
        pair = True

        for i in range(len(ids)):
            if ids[i] == im_id:
                pair_id = i
                break
    return pair, pair_id
