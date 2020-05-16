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
    parser.add_argument("--title", action="store_true")

    args = parser.parse_args()

    data_dir = "./data"
    
    train_data, train_labels, valid_data, valid_labels, test_data = load_data(data_dir)

    use_nn = (args.model == "nn")

    if args.title:
        train_vecs, valid_vecs, test_vecs, t_train_vecs, t_valid_vecs, t_test_vecs = build_corpus(train_data, valid_data, test_data, split_title=True, nn=use_nn)
    else:
        train_vecs, valid_vecs, test_vecs = build_corpus(train_data, valid_data, test_data, nn=use_nn)


    if args.baseline:
        model = build_model(args.model)
        model.train(train_vecs, train_labels, eval_data=valid_vecs, eval_labels=valid_labels)
        rmse, _ = model.eval(valid_vecs, valid_labels)
        print(rmse)
        preds = model.predict(test_vecs)
        output_preds(preds, "results/{}.csv".format(args.model))

        print("Baseline:")
    
    else:
        if args.ensemble == "bagging":
            bagging_hyparam = {
                "model_name": args.model,
                "sample_rate": hyparam["sample_rate"],
                "train_times": hyparam["bagging_train_times"],
                "threshold": hyparam["threshold"],
                "save_path": "models/bagging",
                "output_valid_preds": True
            }
            rmse, test_preds, valid_preds = bagging(bagging_hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=test_vecs)
            if test_preds is not None:
                output_preds(test_preds, "results/bagging_{}.csv".format(args.model))
        elif args.ensemble == "boosting":
            boosting_hyparam = {
                "model_name": args.model,
                "train_times": hyparam["boosting_train_times"],
                "save_path": "models/boosting",
                "output_valid_preds": True
            }
            rmse, test_preds, valid_preds = boosting(boosting_hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=test_vecs)
            # acc, rmse, test_preds = adaboosting(args.model, train_vecs[0:20000], train_labels[0:20000], valid_vecs[0:2000], valid_labels[0:2000], "models/boosting", test_vecs=test_vecs)

            if test_preds is not None:
                output_preds(test_preds, "results/boosting_{}.csv".format(args.model))
        else:
            pass

        # print("Ensemble:")
        # print(rmse)
        a = hyparam["a"]
        bagging_hyparam = {
            "model_name": args.model,
            "sample_rate": hyparam["title_sample_rate"],
            "train_times": hyparam["title_bagging_train_times"],
            "threshold": hyparam["threshold"],
            "save_path": "models/bagging",
            "output_valid_preds": True
        }
        if args.title:
            t_rmse, t_test_preds, t_valid_preds = bagging(bagging_hyparam, t_train_vecs, train_labels, t_valid_vecs, valid_labels, test_vecs=t_test_vecs)
            valid_preds = (a * t_valid_preds + valid_preds) / (a + 1)
            print(t_rmse)
            final_rmse = mean_squared_error(valid_labels, valid_preds) ** 0.5
            print(final_rmse)
        

    print("Hyper params:", hyparam)


if __name__ == "__main__":
    main()
