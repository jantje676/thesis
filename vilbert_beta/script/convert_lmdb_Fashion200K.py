import h5py
import os
from math import floor
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import csv
import base64
import json_lines
import jsonlines
import lmdb # install lmdb by "pip install lmdb"
import pickle
import random
import argparse
import torch.nn as nn
import torch
import _pickle as cPickle

csv.field_size_limit(sys.maxsize)

def main(args):
    random.seed(args.seed)
    data_path = args.data_path
    version = args.version
    clothing = args.clothing
    data_path_out = args.data_path_out
    nr_test = args.nr_test
    n_hard = args.n_hard

    # read features
    features = np.load( "{}/{}/data_ims_{}_train.npy".format(data_path, clothing, version))

    # captions (id, caption)
    captions, img2id = read_captions(data_path, clothing, version)

    save_lmdb(data_path_out, captions, features)

    train_ids = split_data(data_path_out, captions, nr_test, version)

    create_hard_negative(train_ids, features, data_path_out, img2id, n_hard)

def split_data(data_path_out, captions, nr_test, version):
    unique_indices = list(range(0, len(captions)))
    random.shuffle(unique_indices)

    train_indx = unique_indices[nr_test:]
    test_indx = unique_indices[:nr_test]

    train_ids = write_jsonline(captions, train_indx, "train", version, data_path_out)
    _ = write_jsonline(captions, test_indx, "test", version, data_path_out)

    return train_ids

def write_jsonline(captions, indx, split, version, data_path_out):
    image_ids = []
    with jsonlines.open('{}/Gen_{}_{}.jsonline'.format(data_path_out, split, version), mode='w') as writer:
        for i in range(len(indx)):
            writer.write({"sentences": [captions[indx[i]][1]], "id": int(captions[indx[i]][0]), "img_path": captions[indx[i]][0]+ ".jpg"})
            image_ids.append(int(captions[indx[i]][0]))
    return image_ids

def create_hard_negative(train_ids, features, data_path_out, img2id, n_hard):
    # only take the features used in train set
    features_train = np.stack([features[img2id[id]] for id in train_ids], axis=0)

    # take only the last full image
    features_train = torch.from_numpy(features_train[:, 6, :])

    hard_negative = []
    n_features = features_train.shape[0]

    # check n_features en train_ids
    if len(train_ids) != n_features:
        print("not equal")
        exit()

    cos = nn.CosineSimilarity(dim=1)

    for i in range (len(train_ids)):
        # take full image
        feature = features_train[i]
        feature_expand = feature.expand((n_features, -1))
        sims = cos(feature_expand, features_train)
        sims[i] = np.NINF
        results = torch.topk(sims, n_hard)
        hard_negative.append(results[1].tolist())
        if i % 100 == 0:
            print(i)

    negative_pool = np.stack(hard_negative)
    hard_negative = {"train_hard_pool": negative_pool, "train_image_list": train_ids}

    # save pickle {"train_hard_pool" : numpy(29000,100), "train_image_id_list": list(29000) }
    pickle.dump(hard_negative, open('{}/hard_negative.pkl'.format(data_path_out), 'wb'))

def save_lmdb(data_path_out, captions, features):
    id_list = []
    save_path = os.path.join(data_path_out, 'Gen.lmdb')

    count = 0
    env = lmdb.open(save_path, map_size=1099511627776)
    with env.begin(write=True) as txn:

        for i in range(len(captions)):
            img_id = str(captions[i][0]).encode()
            id_list.append(img_id)
            W = 256
            H = 256
            # 1=x_1, 2=y_1, 3=x_2, 4 =y_2 linkerbovenhoek=(x_1, y_1) rechteronderhoek=(x_2, y_2)
            bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                              [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)],
                              [floor(0.5*W),0,W,floor(0.5*H)]], dtype=np.float32)

            # feature = np.float16(features[i])
            feature = features[i]
            item = {
                "image_id": int(img_id),
                "image_h": int(256),
                "image_w": int(256),
                "num_boxes": int(7),
                "boxes" :base64.b64encode(bboxes),
                "features": base64.b64encode(feature),
                "cls_prob" : 0.5
            }

            txn.put(img_id, pickle.dumps(item))
            if count % 1000 == 0:
                print(count)
            count += 1
        txn.put('keys'.encode(), pickle.dumps(id_list))


def read_captions(data_path, clothing, version):
    caption_file = "{}/{}/data_captions_{}_train.txt".format(data_path, clothing, version)
    captions = []
    img2id = {}
    with open(caption_file, newline = '') as file:
        caption_reader = csv.reader(file, delimiter='\t')
        for i ,caption in enumerate(caption_reader):
            img2id[int(caption[0])] = i
            captions.append((caption[0], caption[1]))
    return captions, img2id

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=None, help='version control')
    parser.add_argument('--data_path', default="../../data/Fashion200K", help='path to data folder.')
    parser.add_argument('--data_path_out', default="../data/Fashion200K", help='path to data out folder.')
    parser.add_argument('--seed', default=17, type=int, help='seed')
    parser.add_argument('--clothing', default="dresses", type=str, help='clothing item')
    parser.add_argument('--nr_test', default=3, type=int, help='size of test set')
    parser.add_argument('--n_hard', default=4, type=int, help='size of test set')

    args = parser.parse_args()
    main(args)
