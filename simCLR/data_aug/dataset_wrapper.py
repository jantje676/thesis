import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from torchvision.datasets import ImageFolder
import os
import time

np.random.seed(0)


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

class DataSetWrapper(object):

    def __init__(self,opt):
        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers
        self.valid_size = opt.valid_size
        self.s = opt.s
        self.input_shape = (opt.input_shape_width, opt.input_shape_height)
        self.name_dataset = opt.name_dataset
        self.dset_dir = opt.dset_dir
    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        transform = SimCLRDataTransform(data_augment)

        root = os.path.join(self.dset_dir, 'Fashion200K/{}'.format(self.name_dataset))

        train_kwargs = {"root": root, "transform": transform}
        data = CustomImageFolder(**train_kwargs)

        num_train = len(data)
        splitValid = int(np.floor(self.valid_size * num_train))
        splitTrain = num_train - splitValid

        train_data, val_data = random_split(data, [splitTrain, splitValid])

        train_loader = DataLoader(train_data,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True)

        valid_loader = DataLoader(val_data,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True)
        print(len(train_loader.dataset))
        print(len(valid_loader.dataset))

        # import matplotlib.pyplot as plt
        # nr = 4
        # fig = plt.figure()
        # ax1 = fig.add_subplot(2,2,1)
        # ax1.imshow(train_loader.dataset[nr][0].permute(1, 2, 0) )
        # ax2 = fig.add_subplot(2,2,2)
        # ax2.imshow(train_loader.dataset[nr][1].permute(1, 2, 0) )
        # plt.show()
        # exit()
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(self.input_shape[1], self.input_shape[0])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        end_time = time.time()
        return xi, xj
