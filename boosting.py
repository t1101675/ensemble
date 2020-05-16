import numpy as np
import sklearn
import os
import json

from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from utils import build_model, hyparam

def boosting(hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=None, n_class=5):
    model_name = hyparam["model_name"]
    train_times = hyparam["train_times"]
    output_valid_preds = hyparam["output_valid_preds"]

    train_weights = np.ones(len(train_labels)) / len(train_labels)
    betas = []
    valid_preds = []
    test_preds = []
    rmse = 0
    tbar = tqdm(range(train_times), desc="Boosting Training")
    for e in tbar:
        model = build_model(model_name)
        train_rmse, train_pred = model.train(train_vecs, train_labels, sample_weight=train_weights)
        train_pred = np.array(train_pred)
        wrong_weights = train_weights * (train_pred != train_labels)
        sum_wrong_weight = np.sum(wrong_weights)
        if sum_wrong_weight > 0.5:
            break
            
        beta = sum_wrong_weight / (1 - sum_wrong_weight)

        right_weights = train_weights * (train_pred == train_labels)
        right_weights *= beta
        train_weights = right_weights + wrong_weights
        train_weights /= np.sum(train_weights)
        
        betas.append(beta)
        # model.save(os.path.join(save_path, "boosting-{}-{}".format(model_name, e)))

        valid_rmse, valid_pred = model.eval(valid_vecs, valid_labels)
        valid_preds.append(valid_pred)

        if test_vecs is not None:
            test_pred = model.predict(test_vecs)
            test_preds.append(test_pred)

        # print("train_weights:", train_weights)

    # with open(os.path.join(save_path, "betas.json"), "w") as f:
    #     json.dump(betas, f)

    print("betas: ", betas)

    def boosting_predict(preds, betas):
        boosting_scores = np.zeros((len(preds[0]), n_class))
        for pred, beta in zip(preds, betas):
            for i, p in enumerate(pred):
                boosting_scores[i][p] += np.log(1 / beta)

        boosting_scores /= np.sum(boosting_scores, axis=1)[:, np.newaxis]
        # bagging_preds = np.argmax(bagging_scores, axis=1)
        boosting_preds = np.dot(boosting_scores, np.array(list(range(n_class))))
        return boosting_preds

    v_preds = boosting_predict(valid_preds, betas)
    rmse = mean_squared_error(valid_labels, v_preds) ** 0.5

    t_preds = None
    if test_vecs is not None:
        t_preds = boosting_predict(test_preds, betas)
    
    if output_valid_preds:
        return rmse, t_preds, v_preds
    else:
        return rmse, t_preds

def adaboosting(hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=None):
    v_scores = np.zeros((len(valid_labels), 5))
    t_scores = np.zeros((test_vecs.shape[0], 5))
    for i in range(5):
        train_labels_2 = (train_labels == i) + 0
        valid_labels_2 = (valid_labels == i) + 0
        rmse, t_preds, v_preds = boosting(hyparam, train_vecs, train_labels_2, valid_vecs, valid_labels_2, test_vecs=test_vecs, n_class=2)
        print("class {}, rmse: {}".format(i, rmse))
        v_scores[:, i] = v_preds
        if v_preds is not None:
            t_scores[:, i] = t_preds

    v_scores += 0.000001
    v_scores /= np.sum(v_scores, axis=1)[:, np.newaxis]

    print(v_scores)

    v_preds = np.dot(v_scores, np.array([0, 1, 2, 3, 4]))
    if test_vecs is not None:
        t_scores += 0.000001
        t_scores /= np.sum(t_scores, axis=1)[:, np.newaxis]
        t_preds = np.dot(t_scores, np.array([0, 1, 2, 3, 4]))

    rmse = mean_squared_error(valid_labels, v_preds) ** 0.5

    return rmse, t_preds
