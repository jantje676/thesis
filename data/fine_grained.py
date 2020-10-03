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

    f1_laenen, c_laenen, gamma_laenen, seg_laenen = perform_svm(attributes, data_path, "laenen_1k", test_percentage, min_l)
    f1_layers, c_layers, gamma_layers, seg_layers = perform_svm(attributes, data_path, "layers", test_percentage, min_l)

    print("*********LAENEN*********")
    print("Best f1-score: {}".format(f1_laenen))
    print("c: {}".format(c_laenen))
    print("gamma: {}".format(gamma_laenen))
    print("Segment: {}".format(seg_laenen))

    print("*********LAYERS*********")
    print("Best f1-score: {}".format(f1_layers))
    print("c: {}".format(c_layers))
    print("Layer: {}".format(seg_layers))

def perform_svm(attributes, data_path, version, test_percentage, min_l):
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
    best_c = 0
    best_gamma = 0
    seg = None
    for i in range(7):
        # choose correct fragment
        x_train_seg = x_train[:, i, :]
        x_test_seg = x_test[:, i, :]


        @optunity.cross_validated(x=x_train_seg, y=y_train, num_folds=10, num_iter=2)
        def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
            model = svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
            decision_values = model.decision_function(x_test)
            return optunity.metrics.roc_auc(y_test, decision_values)

        print("{}: Performing hyperparameter tuning layer/segment {}".format(datetime.datetime.now().time(), i))
        hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

        print("Found optimal parameters for layer/seg {}, c({}), gamma({})".format(i ,10 ** hps['logC'],10 ** hps['logGamma'] ))
        # train model on the full training set with tuned hyperparameters
        optimal_model = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(x_train_seg, y_train)
        pred = optimal_model.predict(x_test_seg)
        f1 = calc_score(pred, y_test)

        if f1 > best_f1:
            best_f1 = f1
            best_c = 10 ** hps['logC']
            best_gamma = 10 ** hps['logGamma']
            seg = i

    return best_f1, best_c, best_gamma, seg

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

    parser.add_argument('--attr1',help='which resnet to use', default="black", type=str)
    parser.add_argument('--attr2',help='which resnet to use', default="white", type=str)
    parser.add_argument('--min_l',help='which resnet to use', default=200, type=int)

    args = parser.parse_args()
    main(args)

# create even features
# create train and test set
# train model
# test model
