import os
import time
import shutil
import glob
import json

import torch
import numpy
import tb as tb_logger

import argparse
import numpy as np
import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm




def main(args):
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("find run name")
    args = find_run_name(args)

    batch_size = args.batch_size
    train_split = args.train_split
    pretrained_net = args.pretrained_net
    num_epochs = args.num_epochs
    lr = args.lr
    net = args.net

    print("create logger")
    tb_logger.configure(args.logger_name, flush_secs=5)
    print("save hyperparameters")
    save_hyperparameters(args.logger_name, args)
    transform = get_transform(args.net)

    print("loading data")
    data = ImageFolder(args.image_root, transform=transform)

    train_size = int(len(data) * train_split)
    dev_size = len(data) - train_size

    print("Size train split: {}".format(train_size))
    print("Size dev split: {}".format(dev_size))

    train_data, dev_data = torch.utils.data.random_split(data, [train_size, dev_size ])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=0, pin_memory=True)

    devloader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                          shuffle=False, num_workers=0, pin_memory=True)

    net = get_model(len(data.classes), pretrained_net, net)

    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in  tqdm(range(num_epochs)):
        epochs_no_improve += 1
        print("start training: epoch {}".format(epoch))
        train_loss = train(args, trainloader, net, optimizer, criterion, epoch, device)

        print("start validating: epoch {}".format(epoch))
        loss, acc = validate(args, devloader, net, criterion, tb_logger, epoch, device)

        tb_logger.log_value('Train loss', train_loss, step=epoch)
        tb_logger.log_value('Dev accuracy', acc, step=epoch)
        tb_logger.log_value('Dev loss', loss, step=epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if not os.path.exists(args.model_name):
            os.mkdir(args.model_name)

        last_epoch = False
        if epoch == (args.num_epochs - 1):
            last_epoch = True

        if is_best or last_epoch:
            save_checkpoint({
                'epoch': epoch,
                'model': net.state_dict(),
                'best_loss': best_loss,
                'args': args,
            }, is_best, last_epoch, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=args.model_name + '/')
            epochs_no_improve = 0

        if epochs_no_improve == args.n_epochs_stop:
            print("Early stopping at epoch: {}".format(num_epochs))
            break


def validate(args, devloader, net, criterion, tb_logger, epoch, device):
    net.eval()
    with torch.no_grad():
        correct = 0
        wrong = 0
        running_loss = 0
        for i ,data in enumerate(devloader):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            outputs = net(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            prediction = torch.argmax(outputs,dim=1)
            score = torch.eq(prediction, targets)
            correct_pred = torch.sum(score).item()
            wrong_pred = args.batch_size - correct_pred
            correct += correct_pred
            wrong += wrong_pred
            if i % 100 == 0:
                print("test: [{}, {}/{}] loss: {}".format(epoch, i, len(devloader), loss.item()))

        accuracy = correct/ (wrong + correct)
        print("test accuracy is: {}%".format(accuracy))

    return running_loss, accuracy



def train(args, trainloader, net, optimizer, criterion, epoch, device):
    net.train()
    running_loss = 0
    begin_time = time.time()
    for i ,data in enumerate(trainloader):

        images, targets = data

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 0:
            end_time = (time.time() - begin_time)/100
            print("train: [{}, {}/{}] loss: {} batch_time: {:.3f}".format(epoch, i, len(trainloader), loss.item(), end_time))
            begin_time = time.time()
    return running_loss

def get_transform(net):
    if net == "alex":
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    elif net == "resnet":
        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    return transform

def get_model(n_classes, pretrained_net, net):
    if net == "alex":
        net = models.alexnet(pretrained=pretrained_net)
        net.classifier[6] = nn.Linear(4096, n_classes)
    elif net == "resnet":
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained_net)
        net.fc = nn.Linear(2048, n_classes)

    return net


def save_hyperparameters(log_path, args):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open('{}/commandline_args.txt'.format(log_path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


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

def find_run_name(args):
    folder = "runAlex" if args.net == "alex" else "runRes"
    path = "{}/run*".format(folder)
    runs = glob.glob(path)
    runs.sort()
    nr_next_run = len(runs) + 1
    args.model_name = './{}/run{}/checkpoint'.format(folder,nr_next_run)
    args.logger_name = './{}/run{}/log'.format(folder, nr_next_run)
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='alex',help='alex|resnet')
    parser.add_argument('--image_root', default='../data/Fashion200K/women',help='path to root images')
    parser.add_argument('--train_split', default=0.9, type=float ,help='percentage to take for training')
    parser.add_argument('--num_epochs', default=5, type=int ,help='num epochs')
    parser.add_argument('--lr', default=0.001, type=float ,help='num epochs')
    parser.add_argument('--pretrained_net', action="store_true" ,help='use pretrained net')
    parser.add_argument('--batch_size', default=4, type=int,help='batch size')
    parser.add_argument('--logger_name', default=None, help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default=None,help='Path to save the model.')
    parser.add_argument('--n_epochs_stop', default=5,help='N epochs before early stopping', type=int)


    args = parser.parse_args()
    main(args)
