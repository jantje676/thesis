"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

class padd(object):
    def __call__(self,img):
        W, H = img.size
        # check if image is rectangle shaped
        if H > W:
            diff = H - W
            desired_size = H
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(img, (diff//2, 0))
        elif W > H:
            diff = W - H
            desired_size = W
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(img, (0, diff//2))
        return  new_im


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'


    # need to have all the images in one folder!
    if name.lower() == 'fashion200k':
        root = os.path.join(dset_dir, 'Fashion200K/pictures_only')
        if args.resize ==  "padding":
            transform = transforms.Compose([
                            padd(),
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),])
        elif args.resize == "ratio":
            transform = transforms.Compose([
                            transforms.Resize((args.ratio_width * args.ratio, args.ratio_width)),
                            transforms.ToTensor(),])

        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == 'fashion200k_test':
        root = os.path.join(dset_dir, 'Fashion200K_test/pictures_only')
        if args.resize ==  "padding":
            transform = transforms.Compose([
                            padd(),
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),])
        elif args.resize == "ratio":
            transform = transforms.Compose([
                            transforms.Resize((args.ratio_width * args.ratio, args.ratio_width)),
                            transforms.ToTensor(),])

        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
