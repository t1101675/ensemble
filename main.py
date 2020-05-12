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
hyparam = {
    "threshold": 0,
    "sample_rate": 0.2,
    "bagging_train_times": 100,
    "boosting_train_times": 5,
}


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

    with open(os.path.join(data_dir, "test.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        test_data = [(line[4], line[5]) for line in reader][1:]

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

def bagging(model_name, train_vecs, train_labels, valid_vecs, valid_labels, save_path, test_vecs=None):
    sample_rate = hyparam["sample_rate"]
    train_times = hyparam["bagging_train_times"]
    threshold = hyparam["threshold"]

    valid_preds = []
    test_preds = []
    acc, rmse = 0, 0
    tbar = tqdm(range(train_times), desc="Bagging Training")
    for e in tbar:
        _, samp_vec, _, samp_labels = train_test_split(train_vecs, train_labels, test_size=sample_rate)
        model = build_model(model_name)
        _, _, _ = model.train(samp_vec, samp_labels)
        # model.save(os.path.join(save_path, "bagging-{}-{}.model".format(model_name, e)))
        acc, rmse, pred = model.eval(valid_vecs, valid_labels)
        print({"valid acc": acc, "valid rmse": rmse})
        valid_preds.append(pred)
        if test_vecs is not None:
            pred = model.predict(test_vecs)
            test_preds.append(pred)

    def bagging_predict(preds):
        bagging_scores = np.zeros((len(preds[0]), 5))
        for pred in preds:
            for i, p in enumerate(pred):
                bagging_scores[i][p] += 1

        with open("test.json", "w") as f:
            json.dump(bagging_scores.tolist(), f)
        # tuncate
        bagging_scores *= (bagging_scores >= threshold)
        # normalize
        bagging_scores /= np.sum(bagging_scores, axis=1)[:, np.newaxis]
        # bagging_preds = np.argmax(bagging_scores, axis=1)
        bagging_preds = np.dot(bagging_scores, np.array([0, 1, 2, 3, 4]))
        return bagging_preds

    v_preds = bagging_predict(valid_preds)
    rmse = mean_squared_error(valid_labels, v_preds) ** 0.5
    acc = 0
    # acc = accuracy_score(valid_labels, v_preds)

    t_preds = None
    if test_vecs is not None:
        t_preds = bagging_predict(test_preds)

    return acc, rmse, t_preds


def boosting(model_name, train_vecs, train_labels, valid_vecs, valid_labels, save_path, test_vecs=None):
    train_times = hyparam["boosting_train_times"]
    train_weights = np.ones(len(train_labels)) / len(train_labels)
    tbar = tqdm(range(train_times), desc="Boosting Training")
    betas = []
    valid_preds = []
    test_preds = []
    acc, rmse = 0, 0
    print("Original Train Weights: ", train_weights)
    for e in tbar:
        model = build_model(model_name)
        train_acc, train_rmse, train_pred = model.train(train_vecs, train_labels, sample_weight=train_weights)
        train_pred = np.array(train_pred)
        wrong_weights = train_weights * (train_pred != train_labels)
        sum_wrong_weight = np.sum(wrong_weights)
        if sum_wrong_weight > 0.8:
            break
            
        beta = sum_wrong_weight / (1 - sum_wrong_weight)

        right_weights = train_weights * (train_pred == train_labels)
        right_weights *= beta
        train_weights = right_weights + wrong_weights
        train_weights /= np.sum(train_weights)
        
        betas.append(beta)
        # model.save(os.path.join(save_path, "boosting-{}-{}".format(model_name, e)))

        valid_acc, valid_rmse, valid_pred = model.eval(valid_vecs, valid_labels)
        valid_preds.append(valid_pred)

        if test_vecs is not None:
            test_pred = model.predict(test_vecs)
            test_preds.append(test_pred)

        # print("train_weights:", train_weights)
        print({"train_acc": train_acc, "train_rmse": train_rmse, "valid_acc": valid_acc, "valid_rmse": valid_rmse})

    with open(os.path.join(save_path, "betas.json"), "w") as f:
        json.dump(betas, f)

    print("betas: ", betas)

    def boosting_predict(preds, betas):
        boosting_scores = np.zeros((len(preds[0]), 5))
        for pred, beta in zip(preds, betas):
            for i, p in enumerate(pred):
                boosting_scores[i][p] += np.log(1 / beta)

        boosting_scores /= np.sum(boosting_scores, axis=1)[:, np.newaxis]
        # bagging_preds = np.argmax(bagging_scores, axis=1)
        boosting_preds = np.dot(boosting_scores, np.array([0, 1, 2, 3, 4]))
        return boosting_preds

    v_preds = boosting_predict(valid_preds, betas)
    rmse = mean_squared_error(valid_labels, v_preds) ** 0.5
    # acc = accuracy_score(valid_labels, v_preds)
    acc = 0

    t_preds = None
    if test_vecs is not None:
        t_preds = boosting_predict(test_preds, betas)
    
    return acc, rmse, t_preds

def output_preds(preds, path):
    with open(path, "w") as f:
        f.write("id,predicted\n")
        for i, p in enumerate(preds):
            f.write(str(i + 1) + "," + str(p + 1) + "\n")


def main():
    print("Hyper params: ", hyparam)
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
        output_preds(preds, "results/{}.csv".format(args.model))

        print("Baseline:")
        print(acc)
        print(rmse)
    
    else:
        if args.ensemble == "bagging":
            acc, rmse, test_preds = bagging(args.model, train_vecs, train_labels, valid_vecs, valid_labels, "models/bagging", test_vecs=test_vecs)
            print(len(test_preds))
            if test_preds is not None:
                output_preds(test_preds, "results/bagging_{}.csv".format(args.model))
        elif args.ensemble == "boosting":
            acc, rmse, test_preds = boosting(args.model, train_vecs, train_labels, valid_vecs, valid_labels, "models/boosting", test_vecs=test_vecs)
            if test_preds is not None:
                output_preds(test_preds, "results/boosting_{}.csv".format(args.model))
        else:
            exit(-1)
        print("Ensemble:")
        print(acc)
        print(rmse)

    print("Hyper params:", hyparam)


if __name__ == "__main__":
    main()
