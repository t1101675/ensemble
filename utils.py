import pickle
import json
import sklearn
import numpy as np
import os
import csv

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from models import SVM, DTree

def load_data(data_dir):
    with open(os.path.join(data_dir, "train.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        train_data = [line for line in reader][1:]
        train_labels = [int(float(line[0])) - 1 for line in train_data]
        train_data = [(line[4], line[5]) for line in train_data]
        count = [0, 0, 0, 0, 0]
        for l in train_labels:
            count[l] += 1
        print(count)

    with open(os.path.join(data_dir, "valid.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        valid_data = [line for line in reader][1:]
        valid_labels = [int(float(line[0])) - 1 for line in valid_data]
        valid_data = [(line[4], line[5]) for line in valid_data]

    with open(os.path.join(data_dir, "test.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        test_data = [(line[4], line[5]) for line in reader][1:]

    return train_data, np.array(train_labels), valid_data, np.array(valid_labels), test_data

def _build_corpus(corpus, train_len, valid_len, max_features):
    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    transformer = TfidfTransformer()
    count = vectorizer.fit_transform(corpus)
    all_vecs = transformer.fit_transform(count)
    print("num_features: ", len(vectorizer.get_feature_names()))

    train_vecs = all_vecs[0:train_len]
    valid_vecs = all_vecs[train_len:train_len + valid_len]
    test_vecs = all_vecs[train_len + valid_len:]
    return train_vecs, valid_vecs, test_vecs

def build_corpus(train_data, valid_data, test_data, split_title=False):
    max_features = hyparam["max_features"]
    corpus = []
    print("Preprocessing")
    for line in train_data:
        corpus.append(line[0] + line[1])
    for line in valid_data:
        corpus.append(line[0] + line[1])
    for line in test_data:
        corpus.append(line[0] + line[1])

    train_vecs, valid_vecs, test_vecs = _build_corpus(corpus, len(train_data), len(valid_data), max_features)
    
    if split_title:
        title_max_features = hyparam["title_max_features"]
        corpus = []
        for line in train_data:
            corpus.append(line[0])
        for line in valid_data:
            corpus.append(line[0])
        for line in test_data:
            corpus.append(line[0])

        t_train_vecs, t_valid_vecs, t_test_vecs = _build_corpus(corpus, len(train_data), len(valid_data), title_max_features)
    
    print("Preprocessing end")
    
    if split_title:
        return train_vecs, valid_vecs, test_vecs, t_train_vecs, t_valid_vecs, t_test_vecs
    else:
        return train_vecs, valid_vecs, test_vecs


def build_model(model_name):
    if model_name == "dtree":
        return DTree(max_depth=hyparam["max_depth"])
    elif model_name == "svm":
        return SVM(tol=hyparam["tol"], C=hyparam["C"])
    else:
        print("No model")
        exit(-1)


def output_preds(preds, path):
    with open(path, "w") as f:
        f.write("id,predicted\n")
        for i, p in enumerate(preds):
            f.write(str(i + 1) + "," + str(p + 1) + "\n")


hyparam = {
    "threshold": 0,
    "sample_rate": 0.1,
    "bagging_train_times": 60,
    "boosting_train_times": 20,
    "max_features": 15000,
    "title_max_features": 10000,
    "max_depth": 60,
    "tol": 1e-4,
    "C": 2,
    "a": 0.5
}
