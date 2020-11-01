# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from utils import adap_margin
from stn import STN
from util.layers_model import LayersModel, EncoderImageAttention, LayerAttention
from transformers import BertModel
from cnn_layers import CNN_layers

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, n_attention, n_detectors, pretrained_alex, rectangle, precomp_enc_type='basic',
                 no_imgnorm=False, net="alex"):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """

    if precomp_enc_type == "trans":
        img_enc = STN(n_detectors, embed_size, pretrained_alex, rectangle)
    elif precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == "attention":
        img_enc = EncoderImageAttention(
            img_dim, embed_size, n_attention, no_imgnorm)
    elif precomp_enc_type == "layers":
        img_enc = LayersModel(img_dim, embed_size)
    elif precomp_enc_type == "layers_attention":
        img_enc = LayerAttention(img_dim, embed_size, n_attention, no_imgnorm)
    elif precomp_enc_type == "cnn_layers":
        img_enc = CNN_layers(n_detectors, embed_size, pretrained_alex, net)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


def get_EncoderText(vocab_size, word_dim, embed_size, num_layers, bi_gru, no_txtnorm, txt_enc, vocab_path):
    if txt_enc == "basic":
        text_encoder = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_bi_gru=bi_gru,
                                   no_txtnorm=no_txtnorm)
    elif txt_enc == "word2vec":
        text_encoder = Word2vec(vocab_path, no_txtnorm=no_txtnorm)
    elif txt_enc == "bert":
        text_encoder = Bert(no_txtnorm=no_txtnorm)

    return text_encoder


class Bert(nn.Module):

    def __init__(self,no_txtnorm=False ):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)
        self.no_txtnorm = no_txtnorm

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        batch_size = x.shape[0]
        max_length = x.shape[1]

        segments_ids = self.create_segment_ids(lengths, batch_size, max_length)
        outputs = self.model(x, segments_ids)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        cap_emb = torch.sum(token_embeddings[-4:], dim=0)


        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        cap_len = torch.tensor(lengths, dtype=torch.int)

        return cap_emb, cap_len

    def create_segment_ids(self, lengths, batch_size, max_length):
        segment_ids = torch.zeros((batch_size, max_length))
        for i in range(len(lengths)):
            segment_ids[i, : lengths[i]] = 1
        return segment_ids

class Word2vec(nn.Module):

    def __init__(self, vocab_path, no_txtnorm=False ):
        super(Word2vec, self).__init__()
        weight = torch.load("{}/word2vec.pt".format(vocab_path))
        # word embedding
        self.embed = nn.Embedding.from_pretrained(weight)
        self.no_txtnorm = no_txtnorm

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        cap_emb = self.embed(x)

        cap_len = torch.tensor(lengths, dtype=torch.int)


        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    temp = (w12 / (w1 * w2).clamp(min=eps))

    return temp.squeeze()


def xattn_score_t2i(images, captions, cap_lens, freqs, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    attention_store = []
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        freq = freqs[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)

        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        attention_store.append(attn)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if len(row_sim.shape) == 1:
            row_sim = row_sim.unsqueeze(0)

        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        elif opt.agg_func == "Freq":
            eps = opt.epsilon
            freqs_log = torch.FloatTensor([1 / (np.log(freq[j] + eps)) if freq[j] != 0 else 0 for j in range(len(freq)) ])
            log_sum = torch.sum(freqs_log)
            freqs_log = freqs_log / log_sum
            freqs_log = freqs_log.repeat(n_image,1)
            row_sim = row_sim * freqs_log
            row_sim = row_sim.sum(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities, attention_store


def xattn_score_i2t(images, captions, cap_lens, freqs, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    attention_store = []

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        attention_store.append(attn)

        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if len(row_sim.shape) == 1:
            row_sim = row_sim.unsqueeze(0)

        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        elif opt.agg_func == "freq":
            print("freq approach works only with t2i!!")
            exit()
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities, attention_store


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation


    def forward(self, im, s, s_l, freq_score, freqs, epoch):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores, _ = xattn_score_t2i(im, s, s_l, freqs, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores, _ = xattn_score_i2t(im, s, s_l, freqs, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)


        diagonal = scores.diag().view(im.size(0), 1)

        if self.opt.add_cost == True:
            temp = [1 if score > self.opt.cost_thres else self.opt.gamma for score in freq_score]
            temp = torch.FloatTensor(temp).unsqueeze(dim=1)
            temp  = Variable(temp)
            if torch.cuda.is_available():
                temp = temp.cuda()
            diagonal = diagonal * temp


        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if self.opt.adap_margin:
            margin1, margin2 = adap_margin(freq_score, scores, self.margin)
        else:
            margin1 = self.margin
            margin2 = self.margin


        cost_s = (margin1 + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin2 + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        standard_loss = cost_s.sum() + cost_im.sum()
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # add extra loss function
        if self.opt.diversity_loss:
            num = torch.bmm(im, im.permute(0,2,1))
            norm = torch.norm(im, dim =2).unsqueeze(dim=2)
            denom = torch.bmm(norm, norm.permute(0,2,1))
            sim_im = (num / (denom).clamp(min=1e-08))
            loss_div = torch.triu(sim_im, diagonal=1)
            loss_div = loss_div.sum() * self.opt.theta
            total_loss = standard_loss + loss_div
            diversity_loss = loss_div.item()
        else:
            diversity_loss = 0
            total_loss =  standard_loss



        return total_loss, standard_loss.item(), diversity_loss




class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size, opt.n_attention,
                                    opt.n_detectors, opt.pretrained_alex, opt.rectangle,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm, net=opt.net)

        self.txt_enc = get_EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   opt.bi_gru, opt.no_txtnorm, opt.txt_enc, opt.vocab_path)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
            cudnn.enabled = True


        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)


        if opt.precomp_enc_type == "trans":
            params = list(self.txt_enc.parameters())
            for i in range(opt.n_detectors):
                params += list(self.img_enc.conv[i].parameters())
            # only add learning of spatial transformation when needed
            if not opt.basic:
                params += list(self.img_enc.localization.parameters())
                params += list(self.img_enc.fc_loc.parameters())
        elif opt.precomp_enc_type == "cnn_layers":
            params = list(self.txt_enc.parameters())
            for i in range(opt.n_detectors):
                params += list(self.img_enc.conv[i].parameters())
        elif opt.txt_enc == "bert":
            params =  list(self.img_enc.parameters())
        else:
            params = list(self.txt_enc.parameters())
            params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, epoch, img_emb, cap_emb, cap_len, freq_score, freqs, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        total_loss, standard_loss, loss_div = self.criterion(img_emb, cap_emb, cap_len, freq_score, freqs, epoch)

        self.logger.update('tot_Le', total_loss.item(), img_emb.size(0))
        self.logger.update('stand_Le', standard_loss, img_emb.size(0))
        self.logger.update('div_loss_Le', loss_div, img_emb.size(0))

        return total_loss

    def train_emb(self, epoch, images, captions, lengths, ids=None, freq_score=None, freqs=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_emb, cap_emb, cap_lens, freq_score, freqs)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
