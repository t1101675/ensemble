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

from utils import hyparam, load_data, build_corpus, build_model, output_preds
from bagging import bagging
from boosting import adaboosting, boosting

np.random.seed(888)

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
            # acc, rmse, test_preds = adaboosting(args.model, train_vecs[0:20000], train_labels[0:20000], valid_vecs[0:2000], valid_labels[0:2000], "models/boosting", test_vecs=test_vecs)

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
