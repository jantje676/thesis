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

        c_frag_loss = self.c_frag(sims, cap_l, epoch, same, n_frag, batch_size, n_caption)
        c_glob_loss = self.c_glob(sims, cap_l, image_diag, cap_diag, same, n_frag, batch_size, n_caption)
        loss = c_frag_loss + (0.5 * c_glob_loss)
        return loss


    def c_glob(self, sims, cap_l, image_diag, cap_diag, same, n_frag, batch_size, n_caption):

        sims = sims.sum(dim=[2,3])

        thres_image = get_thres(cap_l, self.n, n_frag).unsqueeze(0).expand(batch_size, -1)
        thres_cap = get_thres(n_frag, self.n, cap_l).unsqueeze(0).expand(batch_size, -1)

        sims_image = sims * thres_image
        sims_cap = sims * thres_cap

        image_diag = image_diag.unsqueeze(1).expand(-1, n_caption)
        cap_diag = cap_diag.unsqueeze(1).expand(-1, n_caption)

        score_image = sims_image - image_diag + self.margin
        score_cap = sims_cap - cap_diag + self.margin

        score_image = self.relu(score_image)
        score_cap = self.relu(score_cap)

        if same:
            score_image.fill_diagonal_(0)
            score_cap.fill_diagonal_(0)

        return score_image.sum() + score_cap.sum()

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


    def sim_pair(self, img_emb_pair, cap_emb_pair, s_l):
        # calculate the similairty score between the image and caption pair
        image_diag, cap_diag = get_sims(img_emb_pair, cap_emb_pair, s_l, self.n)

        return image_diag, cap_diag

def get_sims(img_emb, cap_emb, s_l, n):
    batch_size = cap_emb.size(0)
    cap_l = cap_emb.size(1)
    n_frag = img_emb.size(1)

    # switch axes to use matmul on the 2nd and 3rd dimension
    cap_emb = cap_emb.permute(0,2,1)

    # calculate similarity between the two embeddings
    sim_cap = torch.matmul(img_emb , cap_emb)

    # sum over the dimensions
    sim_cap = sim_cap.sum(dim=[1,2])

    # find threshold values
    thres_image = get_thres(s_l, n, n_frag)
    thres_cap = get_thres(n_frag, n, s_l)

    # calculate sim_pair
    cap_diag = sim_cap * thres_cap
    image_diag = sim_cap * thres_image

    return image_diag, cap_diag

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
