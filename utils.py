import pickle
import json
import sklearn
import numpy as np
import os
import csv
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from models import SVM, DTree, NN

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


def tokenize(text):
    text = re.sub(
        "[+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）~]+", " ", text)
    text = text.replace('\n', ' ')
    text = text.strip().split(' ')
    text = [w.lower() for w in text if w]
    return text

def _build_corpus_nn(corpus, train_len, valid_len):
    all_words = []
    for text in corpus:
        all_words.extend(tokenize(text))

    vocab = set(all_words)
    vocab = list(vocab)
    hyparam["vocab_size"] = len(vocab) + 1
    print("vocab_size: ", len(vocab) + 1)
    w2id = {w:i+1 for i, w in enumerate(vocab)}
    all_vecs = [[w2id[w] for w in tokenize(text)] for text in corpus]

    train_vecs = all_vecs[0:train_len]
    valid_vecs = all_vecs[train_len:train_len + valid_len]
    test_vecs = all_vecs[train_len + valid_len:]

    return train_vecs, valid_vecs, test_vecs

def build_corpus(train_data, valid_data, test_data, split_title=False, nn=False):
    max_features = hyparam["max_features"]
    corpus = []
    print("Preprocessing")
    for line in train_data:
        corpus.append(line[0] + line[1])
    for line in valid_data:
        corpus.append(line[0] + line[1])
    for line in test_data:
        corpus.append(line[0] + line[1])

    if nn:
        train_vecs, valid_vecs, test_vecs = _build_corpus_nn(corpus, len(train_data), len(valid_data))
    else:
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
    elif model_name == "nn":
        return NN(hyparam["vocab_size"], hyparam["n_embd"], hyparam["n_hidden"], hyparam["train_batch_size"], hyparam["valid_batch_size"], hyparam["device"], hyparam["epoch"])
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
    "sample_rate": 0.2,
    "title_sample_rate": 0.2,
    "title_bagging_train_times": 200,
    "bagging_train_times": 60,
    "boosting_train_times": 20,
    "max_features": 15000,
    "title_max_features": 15000,
    "max_depth": 60,
    "tol": 1e-4,
    "C": 2,
    "a": 0.3,
    "n_embd": 256,
    "n_hidden": 256,
    "train_batch_size": 256,
    "valid_batch_size": 8,
    "device": "cuda",
    "epoch": 200,
    "vocab_size": -1
}
