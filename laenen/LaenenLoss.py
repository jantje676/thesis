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
        self.margin = 40

    def forward(self, epoch, img_emb, cap_emb, cap_l, image_diag, cap_diag, same):
        n_frag = img_emb.size(1)
        batch_size = img_emb.size(0)
        n_caption = cap_emb.size(0)

        sims = torch.einsum('bik,ljk->blij', img_emb, cap_emb)

        # c_frag_loss = self.c_frag(sims, cap_l, epoch, same, n_frag, batch_size, n_caption)
        c_glob_loss = self.c_glob2(sims, cap_l, image_diag, cap_diag, same, n_frag, batch_size, n_caption)
        loss = c_glob_loss
        return loss

    def c_glob2(self, sims, cap_l, image_diag, cap_diag, same, n_frag, batch_size, n_caption):
        sims, _ = torch.max(sims, dim=2)

        sims = sims.sum(dim=2)

        thres_image = get_thres2(cap_l, self.n).unsqueeze(0).expand(batch_size, -1)

        sims = sims * thres_image

        image_diag = image_diag.unsqueeze(1).expand(-1, n_caption)
        cap_diag = cap_diag.unsqueeze(0).expand(batch_size, -1)

        score_image = sims - image_diag + self.margin
        score_cap = sims - cap_diag + self.margin

        score_image = self.relu(score_image)
        score_cap = self.relu(score_cap)

        if same:
            score_image.fill_diagonal_(0)
            score_cap.fill_diagonal_(0)

        return score_cap.sum()+ score_image.sum()

    def sim_val(self, img, cap, l):
        batch_size = img.size(0)
        sims = torch.einsum('bik,ljk->blij', img, cap)

        sims, _ = torch.max(sims, dim=2)

        sims = sims.sum(dim=2)

        thres_image = get_thres2(l, self.n).unsqueeze(0).expand(batch_size, -1)

        sims = sims * thres_image

        return sims

    def c_frag(self, sims, cap_l, epoch, same, n_frag, batch_size, n_caption):
        loss = 0

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_l[i]

            sims_i = sims[:,i,:,:n_word]

            # first n epochs fix the constants y_ij
            if epoch < self.switch:
                y_i = init_y(sims_i, same, i)
            # after let the model optimize y_ij with the heuristic sign
            else:
                y_i = sign(sims_i, same, i)

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
        thres = get_thres2(s_l, self.n)

        # calculate sim_pair
        diag = sims* thres

        return diag

def get_thres2(l, n):
    thres = (l + n)
    thres = float(1)/thres.to(dtype=torch.float)

    return thres

def get_thres(a, n, b):
    thres = (a + n) * b
    thres = float(1)/thres.to(dtype=torch.float)

    return thres

def sign(sims_i, same, i):
    y = torch.sign(sims_i)
    n_frag = sims_i.size(1)
    n_word = sims_i.size(2)

    if same:
        temp_y = y[i, :, :]
        # check if at least one in every row has positive sign
        temp_sum = temp_y.sum(dim=0)
        sign_check = temp_sum > (n_frag * -1)
        if sign_check.sum() != n_word:
            for j in range(len(sign_check)):
                if sign_check[j] == False:
                    i_max = torch.argmax(sims_i[i,:,j])
                    y[i, i_max, j] = 1

    return y

# init y matrix with ones when image and word fragment are from the same pair
def init_y(sims_i, same, i):
    y =  torch.ones(sims_i.shape, requires_grad=True) * -1
    if same:
        y[i, :, :] = 1
    return y
