import argparse
import logging
import pickle
import json
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

    return train_data, np.array(train_labels), valid_data, np.array(valid_labels), test_data

def build_corpus(train_data, valid_data, test_data):
    corpus = []
    print("Preprocessing")
    for line in train_data:
        corpus.append(line[0] + line[1])
    for line in valid_data:
        corpus.append(line[0] + line[1])
    for line in test_data:
        corpus.append(line[0] + line[1])

    vectorizer = CountVectorizer(stop_words="english", max_features=20000)
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
    sample_rate = 0.2
    train_times = 10
    preds = []
    bagging_scores = np.zeros((len(valid_labels), 5))
    acc, rmse = 0, 0
    tbar = tqdm(range(train_times), desc="Bagging Training")
    for e in tbar:
        _, samp_vec, _, samp_labels = train_test_split(train_vecs, train_labels, test_size=sample_rate)
        model = build_model(model_name)
        _, _, _ = model.train(samp_vec, samp_labels)
        print({"acc": acc, "rmse": rmse})
        model.save(os.path.join(save_path, "bagging-{}-{}.model".format(model_name, e)))
        acc, rmse, pred = model.eval(valid_vecs, valid_labels)
        preds.append(pred)

    for pred in preds:
        for i, p in enumerate(pred):
            bagging_scores[i][p] += 1

    bagging_preds = np.argmax(bagging_scores, axis=1)

    rmse = mean_squared_error(valid_labels, bagging_preds) ** 0.5
    acc = accuracy_score(valid_labels, bagging_preds)
    return acc, rmse


def boosting(model_name, train_vecs, train_labels, valid_vecs, valid_labels, save_path):
    train_times = 5
    train_weights = np.ones(len(train_labels)) / len(train_labels)
    tbar = tqdm(range(train_times), desc="Boosting Training")
    betas = []
    preds = []
    acc, rmse = 0, 0
    boosting_scores = np.zeros((len(valid_labels), 5))
    print("Original Train Weights: ", train_weights)
    for e in tbar:
        model = build_model(model_name)
        train_acc, train_rmse, train_pred = model.train(train_vecs, train_labels, sample_weight=train_weights)
        train_pred = np.array(train_pred)
        wrong_weights = train_weights * (train_pred != train_labels)
        sum_wrong_weight = np.sum(wrong_weights)
        if sum_wrong_weight > 0.2:
            break
            
        beta = sum_wrong_weight / (1 - sum_wrong_weight)

        right_weights = train_weights * (train_pred == train_labels)
        right_weights *= beta
        train_weights = right_weights + wrong_weights
        train_weights /= np.sum(train_weights)
        
        betas.append(beta)
        model.save(os.path.join(save_path, "boosting-{}-{}".format(model_name, e)))

        valid_acc, valid_rmse, valid_pred = model.eval(valid_vecs, valid_labels)
        preds.append(valid_pred)

        print("train_weights:", train_weights)
        print({"train_acc": train_acc, "train_rmse": train_rmse, "valid_acc": valid_acc, "valid_rmse": valid_rmse})

    with open(os.path.join(save_path, "betas.json"), "w") as f:
        json.dump(betas, f)

    print("betas: ", betas)

    for pred, beta in zip(preds, betas):
        for i, p in enumerate(pred):
            boosting_scores[i][p] += np.log(1 / beta)

    boosting_preds = np.argmax(boosting_scores, axis=1)

    rmse = mean_squared_error(valid_labels, boosting_preds) ** 0.5
    acc = accuracy_score(valid_labels, boosting_preds)
    return acc, rmse

def output_preds(preds, path):
    with open(path, "w") as f:
        f.write("id,predicted\n")
        for i, p in enumerate(preds):
            f.write(str(i + 1) + "," + str(p + 1) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="svm")
    parser.add_argument("--ensemble", type=str, default="bagging")
    parser.add_argument("--baseline", action="store_true")

    args = parser.parse_args()

    data_dir = "./data"
    
    train_data, train_labels, valid_data, valid_labels, test_data = load_data(data_dir)

    train_vecs, valid_vecs, test_vecs = build_corpus(train_data, valid_data, test_data)

    if args.baseline:
        model = build_model(args.model)
        model.train(train_vecs, train_labels)
        acc, rmse, _ = model.eval(valid_vecs, valid_labels)
        preds = model.predict(test_vecs)

        print("Baseline:")
        print(acc)
        print(rmse)
        output_preds(preds, "results/predicts.csv")
    
    else:
        if args.ensemble == "bagging":
            acc, rmse = bagging(args.model, train_vecs, train_labels, valid_vecs, valid_labels, "models/bagging")
        elif args.ensemble == "boosting":
            acc, rmse = boosting(args.model, train_vecs, train_labels, valid_vecs, valid_labels, "models/boosting")
        else:
            exit(-1)
        print("Ensemble:")
        print(acc)
        print(rmse)


if __name__ == "__main__":
    main()
