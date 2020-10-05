import numpy as np
from sklearn import svm
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import optunity
import optunity.metrics
import argparse
import datetime

def main(args):
    attributes = ["white", "black"]
    data_path = "Fashion200K/dresses"
    test_percentage = 0.1

    attributes = (args.attr1, args.attr2)
    min_l = args.min_l

    f1_laenen, hps_laenen, seg_laenen = perform_optim(attributes, data_path, "laenen_1k", test_percentage, min_l, args.svm_lin)
    f1_layers, hps_layers, seg_layers = perform_optim(attributes, data_path, "layers", test_percentage, min_l, args.svm_lin)

    print("*********LAENEN*********")
    print("Best f1-score: {}".format(f1_laenen))
    for key in hps_laenen.keys():
        print("{}: {}".format(key, 10 ** hps_laenen[key]))
    print("Segment: {}".format(seg_laenen))

    print("*********LAYERS*********")
    print("Best f1-score: {}".format(f1_layers))
    for key in hps_layers.keys():
        print("{}: {}".format(key, 10 ** hps_layers[key]))
    print("Layer: {}".format(seg_layers))

def perform_optim(attributes, data_path, version, test_percentage, min_l, svm_lin):
    # read features and captions
    features, captions = read_data(version, data_path)

    # filter for requiered words
    features1 = filter(attributes[0], features, captions)
    features2 = filter(attributes[1], features, captions)

    # find minimum length
    min_nr = min(len(features1), len(features2))

    if min_nr < min_l:
        min_l = min_nr

    print("Total data size is {}".format(min_l * 2))

    # make both sets equal
    features1 = random.sample(features1, min_l)
    features2 = random.sample(features2, min_l)

    # create test_set
    x_train, x_test, y_train, y_test = create_train_test(features1, features2, test_percentage)

    best_f1 = 0
    best_hps = None
    seg = None

    for i in range(7):
        # choose correct fragment
        x_train_seg = x_train[:, i, :]
        x_test_seg = x_test[:, i, :]


        @optunity.cross_validated(x=x_train_seg, y=y_train, num_folds=5, num_iter=1)
        def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
            model = svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
            decision_values = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, decision_values)


        @optunity.cross_validated(x=x_train_seg, y=y_train, num_folds=5, num_iter=1)
        def svml_auc(x_train, y_train, x_test, y_test, logC):
            model = svm.LinearSVC(C=10 ** logC).fit(x_train, y_train)
            decision_values = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, decision_values)

        print("{}: Performing hyperparameter tuning layer/segment {}".format(datetime.datetime.now().time(), i))
        if svm_lin:
            hps, _, _ = optunity.maximize(svml_auc, num_evals=100, logC=[-5, 2])
        else:
            hps, _, _ = optunity.maximize(svm_auc, num_evals=100, logC=[-5, 2], logGamma=[-5, 1])

        print("Found optimal parameters for layer/seg {}, {}".format(i , hps ))
        # train model on the full training set with tuned hyperparameters
        if svm_lin:
            optimal_model = svm.LinearSVC(C=10 ** hps['logC']).fit(x_train_seg, y_train)
        else:
            optimal_model = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(x_train_seg, y_train)

        pred = optimal_model.predict(x_test_seg)
        f1 = calc_score(pred, y_test)

        if f1 > best_f1:
            best_f1 = f1
            best_hps = hps
            seg = i

    return best_f1, best_hps, seg


def calc_score(pred, y_test):
    score = (pred == y_test)
    f1 = f1_score(y_test, pred, average='binary')
    print(pred)
    print(y_test)
    print("Total number of predictions {}".format(len(pred)))
    print("Percentage of correct predictions {}%".format(score.sum()/len(pred)*100))
    print("f1 score: {}".format(f1))
    return f1

def create_train_test(features1, features2, test_percentage):
    n_test = math.ceil(len(features1) * test_percentage)
    test1 = features1[:n_test]
    train1 = features1[n_test:]

    test2 = features2[:n_test]
    train2 = features2[n_test:]

    test = test1 + test2
    train = train1 + train2

    y_test = np.concatenate((np.zeros(len(test1)), np.ones(len(test2))), axis=0)
    y_train = np.concatenate((np.zeros(len(train1)),np.ones(len(train2))), axis=0)

    train, y_train = shuffle(train, y_train, random_state=0)
    test, y_test = shuffle(test, y_test, random_state=0)

    train = np.stack(train)
    test = np.stack(test)
    return train, test, y_train, y_test



def filter(word, features, captions):
    filtered = []
    for i in range(len(captions)):
        temp = captions[i].split("\t")
        caption = temp[1]
        if word in caption.split(" "):
            filtered.append(features[i])
    return filtered

def read_data(version, path):
    file = open("{}/data_captions_{}_train.txt".format(path, version),"r")
    captions = file.readlines()
    features = np.load( "{}/data_ims_{}_train.npy".format(path, version))
    return features, captions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM")

    parser.add_argument('--attr1',help='filter for attribute nr one', default="black", type=str)
    parser.add_argument('--attr2',help='filter for attribute nr two', default="white", type=str)
    parser.add_argument('--min_l',help='maximum nr of features for one word', default=500, type=int)
    parser.add_argument('--svm_lin', action='store_true', help="use linear svm instead of rbf kerner")
    parser.add_argument('--data_path',help='path to data', default="Fashion200K/dresses", type=str)


    args = parser.parse_args()
    main(args)

# create even features
# create train and test set
# train model
# test model
