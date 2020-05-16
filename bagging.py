import json
import sklearn
import numpy as np
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import build_model

def bagging(hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=None):
    sample_rate = hyparam["sample_rate"]
    train_times = hyparam["train_times"]
    threshold = hyparam["threshold"]
    model_name = hyparam["model_name"]
    output_valid_preds = hyparam["output_valid_preds"]

    valid_preds = []
    test_preds = []
    rmse = 0
    tbar = tqdm(range(train_times), desc="Bagging Training")
    for e in tbar:
        _, samp_vec, _, samp_labels = train_test_split(
            train_vecs, train_labels, test_size=sample_rate)
        model = build_model(model_name)
        _, _, _ = model.train(samp_vec, samp_labels)
        # model.save(os.path.join(save_path, "bagging-{}-{}.model".format(model_name, e)))
        rmse, pred = model.eval(valid_vecs, valid_labels)
        print({"valid rmse": rmse})
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

    t_preds = None
    if test_vecs is not None:
        t_preds = bagging_predict(test_preds)

    if output_valid_preds:
        return rmse, t_preds, v_preds
    else:
        return rmse, t_preds
