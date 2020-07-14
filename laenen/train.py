# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""
"BEFORE RUNNING"
"generate_tsv_ken.py"
"split_data.py"
"vocab.py"
import os
import time
import shutil


import torch
import numpy

import data_ken
from vocab import Vocabulary, deserialize_vocab
from model_laenen import SCAN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i
from torch.autograd import Variable
from utils import save_hyperparameters
import logging
import tb as tb_logger
import numpy as np
import random

def start_experiment(opt, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("Number threads:" , torch.get_num_threads())

    # Load Vocabulary Wrapper, create dictionary that can switch between ids and words
    vocab = deserialize_vocab("{}/{}/{}_vocab_{}.json".format(opt.vocab_path, opt.clothing, opt.data_name, opt.version))

    opt.vocab_size = len(vocab)

    # Load data loaders
    first_loader, second_loader, val_loader = data_ken.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt)

    # save hyperparameters in file
    save_hyperparameters(opt.logger_name, opt)

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, model, epoch, first_loader, second_loader, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)

        last_epoch = False
        if epoch == (opt.num_epochs - 1):
            last_epoch = True

        # only save when best epoch, or last epoch for further training
        if is_best or last_epoch:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, last_epoch, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
    return best_rsum

def train(opt, model, epoch, first_loader, second_loader, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for j, first_data in enumerate(first_loader):
        # switch to train mode
        model.train_start()

        for i, second_data in enumerate(second_loader):
            if j == i:
                same = True
            else:
                same = False

            # measure data loading time
            data_time.update(time.time() - end)

            # make sure train logger is used
            model.logger = train_logger

            # Update the model
            model.train_emb(epoch, first_data, second_data, same)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                        epoch, i + (j * len(first_loader)), len(second_loader) * len(second_loader), batch_time=batch_time,
                        data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

            # validate at every val_step
            if model.Eiters % opt.val_step == 0:
                validate(opt, val_loader, model)

# see how well the model makes captions
def validate(opt, val_loader, model):

    # compute the encoding for all the validation images and captions
    with torch.no_grad():
        img_embs, cap_embs, cap_lens = encode_data(
            model, val_loader, opt.log_step, logging.info)

        img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 1)])
        start = time.time()

        # find the similarity between every caption and image in the validation set?

        sims = shard_xattn_t2i(model, img_embs, cap_embs, cap_lens, opt, shard_size=opt.shard_size)

        end = time.time()
        print("calculate similarity time:", end-start)

        # caption retrieval (find the right text with every image)
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
        logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1, r5, r10, medr, meanr))
        # image retrieval (find the right image for every text)
        (r1i, r5i, r10i, medri, meanr) = t2i(
            img_embs, cap_embs, cap_lens, sims)
        logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                     (r1i, r5i, r10i, medri, meanr))
        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, last_epoch, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if is_best:
                torch.save(state, prefix + 'model_best.pth.tar')
            elif last_epoch:
                torch.save(state, prefix + filename)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
