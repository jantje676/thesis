import csv
import numpy as np
from math import floor

train_split = 0.7
dev_split = 0.15
test_split = 0.15


file = open("../../data/Fashion200K/data_captions.txt","r")
captions = file.readlines()
file.close()



features = np.load( "../../data/Fashion200K/data_ims.npy")


caption_count = len(captions) # fileObject is your csv.reader


if features.shape[0] != caption_count:
    print("Shapes do not match!")
    exit()

train = floor(train_split * caption_count)
dev = floor(train_split * caption_count) + floor(dev_split * caption_count)
test = caption_count



if train + (dev - train) + (test - dev) != caption_count:
    print("Lenghts do not match!")
    exit()

with open('../../data/Fashion200K/data_captions_train.txt', 'w', newline='') as file:

    for id in captions[:train]:
        file.write(id)


with open('../../data/Fashion200K/data_captions_dev.txt', 'w', newline='') as file:
    for id in captions[train:dev]:
        file.write(id)

with open('../../data/Fashion200K/data_captions_test.txt', 'w', newline='') as file:
    for id in captions[dev:test]:
        file.write(id)

features_train = features[:train]
features_dev = features[train:dev]
features_test = features[dev:test]

np.save( "../../data/Fashion200K/data_ims_train.npy", features_train)
np.save( "../../data/Fashion200K/data_ims_dev.npy", features_dev)
np.save( "../../data/Fashion200K/data_ims_test.npy", features_test)
