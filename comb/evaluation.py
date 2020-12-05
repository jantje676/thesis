# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os

import sys
from data_ken import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import SCAN, xattn_score_t2i, xattn_score_i2t
from collections import OrderedDict
import time
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """

        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    freqs_new = None

    max_n_word = 0
    for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids, freq_score, freqs) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=True)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            freqs_new = [0] * len(data_loader.dataset)
        # cache embeddings
        # changes ids tuple to list
        ids = list(ids)

        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
            freqs_new[nid] = freqs[j]

        # measure accuracy and record loss, first argument is 100 for LaenenLoss
        model.forward_loss(100 ,img_emb, cap_emb, cap_len, freq_score, freqs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs, cap_lens, freqs_new



def evalrank(model_path,run, data_path=None, split='dev', fold5=False, vocab_path="../vocab/", change=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)

    # add because div_transform is not present in model
    d = vars(opt)
    d['tanh'] = False


    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab("{}{}/{}_vocab_{}.json".format(vocab_path, opt.clothing, opt.data_name, opt.version))
    opt.vocab_size = len(vocab)
    print(opt.vocab_size)
    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    if change:
        opt.clothing = "dresses"

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens, freqs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] , cap_embs.shape[0]))

    t2i_switch = True
    if opt.cross_attn == 't2i':
        sims, attn = shard_xattn_t2i(img_embs, cap_embs, cap_lens, freqs, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims, attn = shard_xattn_i2t(img_embs, cap_embs, cap_lens, freqs, opt, shard_size=128)
        t2i_switch = False
    else:
        raise NotImplementedError

    # r = (r1, r2, r5, medr, meanr), rt= (ranks, top1)
    r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % ri)

    if opt.trans:
        save_dir = "plots_trans"
    else:
        save_dir = "plots_scan"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({'rt': rt, 'rti': rti, "attn": attn, "t2i_switch": t2i_switch }, '{}/ranks_{}_{}.pth.tar'.format(save_dir,run, opt.version))
    return rt, rti, attn, r, ri


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def shard_xattn_t2i(images, captions, caplens, freqs, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """

    # takes a block and calculated the similarity, instead of entire d-matrix in one time
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1

    d = np.zeros((len(images), len(captions)))

    attention = []


    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(int(n_cap_shard)):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            if torch.cuda.is_available():
                im = Variable(torch.from_numpy(images[im_start:im_end])).cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
            else:
                im = Variable(torch.from_numpy(images[im_start:im_end]))
                s = Variable(torch.from_numpy(captions[cap_start:cap_end]))
            l = caplens[cap_start:cap_end]
            f = freqs[cap_start:cap_end]

            sim, attn = xattn_score_t2i(im, s, l, f, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

            attention = attention + attn
    sys.stdout.write('\n')
    return d, attention


def shard_xattn_i2t(images, captions, caplens, freqs, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1

    attention = []
    d = np.zeros((len(images), len(captions)))
    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(int(n_cap_shard)):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            if torch.cuda.is_available():
                im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            else:
                im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True)
                s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True)
            l = caplens[cap_start:cap_end]
            sim, attn = xattn_score_i2t(im, s, l, freqs, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
            attention = attention + attn
    sys.stdout.write('\n')
    return d, attention


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    top1: highest ranking caption/image for every id
    rank: what is the rank of the corresponding image/caption,
          when 0 the image-caption is matched, because rank is highest
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):

        # sort the array and reverse the order to find most similar caption to image
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20

        # return indexes where inds == index
        tmp = np.where(inds == index)
        tmp = tmp[0][0]

        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10,r20, r50, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):

        inds = np.argsort(sims[index])[::-1]
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, medr, meanr)
