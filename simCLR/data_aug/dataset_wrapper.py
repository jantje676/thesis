import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from torchvision.datasets import ImageFolder
import os

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

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, name_dataset, dset_dir):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.name_dataset = name_dataset
        self.dset_dir = dset_dir
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
                                  shuffle=False)

        valid_loader = DataLoader(val_data,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=False)
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

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
