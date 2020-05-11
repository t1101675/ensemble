import argparse
import logging
import pickle
import sklearn
import numpy as np
import os
import csv

from models import DTree, SVM
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

np.random.seed(888)

def load_data(data_dir):
    with open(os.path.join(data_dir, "train.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        train_data = [line for line in reader][1:]
        train_labels = [int(float(line[0])) - 1 for line in train_data]
        train_data = [(line[4], line[5]) for line in train_data]

    with open(os.path.join(data_dir, "valid.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        valid_data = [line for line in reader][1:]
        valid_labels = [int(float(line[0])) - 1 for line in valid_data]
        valid_data = [(line[4], line[5]) for line in valid_data]

        # print(valid_data)
        # exit(0)

    with open(os.path.join(data_dir, "test.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        test_data = [(line[4], line[5]) for line in reader][1:]
    # print(train_data[0])
    # print(valid_data[0])
    # print(test_data[0])
    # exit(0)

    return train_data, train_labels, valid_data, valid_labels, test_data

def build_corpus(train_data, valid_data, test_data):
    corpus = []
    print("Preprocessing")
    for line in train_data:
        corpus.append(line[0] + line[1])
    for line in valid_data:
        corpus.append(line[0] + line[1])
    for line in test_data:
        corpus.append(line[0] + line[1])

    vectorizer = CountVectorizer(stop_words="english", max_features=10000)
    transformer = TfidfTransformer()
    count = vectorizer.fit_transform(corpus)
    tfidf_matrix = transformer.fit_transform(count)
    
    print(len(vectorizer.get_feature_names()))

    train_vecs = tfidf_matrix[0:len(train_data)]
    valid_vecs = tfidf_matrix[len(train_data):len(train_data) + len(valid_data)]
    test_vecs = tfidf_matrix[len(train_data) + len(valid_data):]
    print("Preprocessing end")
    
    return train_vecs, valid_vecs, test_vecs

def build_model(model_name):
    if model_name == "dtree":
        return DTree()
    elif model_name == "svm":
        return SVM()
    else:
        print("No model")
        exit(-1)

def bagging(model_name, train_vecs, train_labels, valid_vecs, valid_labels, save_path):
    sample_rate = 0.1
    train_times = 50
    preds = []
    bagging_scores = np.zeros((len(valid_labels), 5))
    acc, rmse = 0, 0
    tbar = tqdm(range(train_times), desc="Bagging Training")
    for e in tbar:
        _, samp_vec, _, samp_labels = train_test_split(train_vecs, train_labels, test_size=sample_rate)
        model = build_model(model_name)
        acc, rmse = model.train(samp_vec, samp_labels)
        tbar.set_postfix({"acc": acc, "rmse": rmse})
        model.save(os.path.join(save_path, "dtree-{}.model".format(e)))
        pred = model.predict(valid_vecs)
        preds.append(pred)

    for pred in preds:
        for i, p in enumerate(pred):
            bagging_scores[i][p] += 1

    bagging_preds = np.argmax(bagging_scores, axis=1)

    rmse = mean_squared_error(valid_labels, bagging_preds) ** 0.5
    acc = accuracy_score(valid_labels, bagging_preds)
    return acc, rmse

def boosting(model_name, train_vecs, train_labels, valid_vecs, valid_labels, save_path):
    pass

def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    data_dir = "./data"
    
    train_data, train_labels, valid_data, valid_labels, test_data = load_data(data_dir)

    train_vecs, valid_vecs, test_vecs = build_corpus(train_data, valid_data, test_data)
    # print(train_vecs[1])
    # print(train_data[1])
    # exit(0)
    # svm = SVM()

    # train_acc, train_rmse = svm.train(train_vecs.toarray(), train_labels)
    # valid_acc, valid_rmse = svm.eval(valid_vecs.toarray(), valid_labels)

    # print(train_acc, train_rmse)
    # print(valid_acc, valid_rmse)
    acc, rmse = bagging("svm", train_vecs, train_labels, valid_vecs, valid_labels, "models/")

    print("Final")
    print(acc)
    print(rmse)

if __name__ == "__main__":
    main()
