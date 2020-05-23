from bagging import bagging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle

class BaggingNetDataset(Dataset):
    def __init__(self, vecs, scores, labels):
        super().__init__()
        self.vecs = vecs
        self.scores = scores
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        vec = torch.tensor(self.vecs[item].toarray(), dtype=torch.float).squeeze(0)
        # print([s[item] for s in self.scores])
        tmp_score = torch.tensor([s[item] for s in self.scores], dtype=torch.long)
        score = torch.zeros(tmp_score.size(0), 5).scatter(1, tmp_score.unsqueeze(1), 1.0)
        # print(score)
        # exit(0)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        # label = torch.zeros(5)
        # label[self.labels[item]] = 1.0

        return vec, score, label


class BaggingNet(nn.Module):
    def __init__(self, n_embed, n_output, n_hidden):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)
        self.active_fct = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, vecs, scores, labels):
        x = self.linear1(vecs)
        x = self.active_fct(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        weights = F.softmax(logits, dim=1)
        print(weights[0])
        s = torch.matmul(weights.unsqueeze(1), scores).squeeze(1)
        s = s.matmul(torch.tensor([0.0, 1, 2, 3, 4], device="cuda"))
        if labels is not None:
            loss = F.mse_loss(s, labels)
            return s, loss
        else:
            return s


def bagging_net(hyparam, train_vecs, train_labels, valid_vecs, valid_labels, test_vecs=None):
    hyparam["bagging_hyparam"]["output_all"] = True
    batch_size = hyparam["batch_size"]
    eval_batch_size = hyparam["eval_batch_size"]
    n_embed = hyparam["n_embed"]
    n_output = hyparam["n_output"]
    n_hidden = hyparam["n_hidden"]
    epoch = hyparam["epoch"]
    lr = hyparam["lr"]
    device = torch.device(hyparam["device"])

    try:
        with open("train_preds.pkl", "rb") as f:
            train_preds = pickle.load(f)
        with open("valid_preds.pkl", "rb") as f:
            valid_preds = pickle.load(f)
        with open("test_preds.pkl", "rb") as f:
            test_preds = pickle.load(f)
    except:
        _, _, _, train_preds, valid_preds, test_preds = bagging(hyparam["bagging_hyparam"], train_vecs, train_labels, valid_vecs, valid_labels, test_vecs)
        with open("train_preds.pkl", "wb") as f:
            pickle.dump(train_preds, f)
        with open("valid_preds.pkl", "wb") as f:
            pickle.dump(valid_preds, f)
        with open("test_preds.pkl", "wb") as f:
            pickle.dump(test_preds, f)

    model = BaggingNet(n_embed, n_output, n_hidden)
    model = model.to(device)
    train_dataset = BaggingNetDataset(train_vecs, train_preds, train_labels)
    sampler = RandomSampler(train_dataset)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)

    def evaluate():
        eval_dataset = BaggingNetDataset(valid_vecs, valid_preds, valid_labels)
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, sampler=sampler)
        
        total_steps = 0
        total_loss = 0
        preds = []
        for batch in tqdm(data_loader, desc="Bagging NN Evaling"):
            vecs, scores, labels = batch
            vecs = vecs.to(device)
            scores = scores.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred, loss = model(vecs, scores, labels)
                total_steps += 1
                total_loss += loss.item()

                preds.extend(pred.to(torch.device("cpu")).numpy().tolist())

        # preds = [np.dot(pred, np.array([0, 1, 2, 3, 4])) for pred in preds]        
        rmse = mean_squared_error(valid_labels, preds) ** 0.5
        return rmse, preds

    total_steps = 0
    total_loss = 0

    for e in tqdm(range(epoch)):
        model.train()
        for batch in tqdm(data_loader, desc="Bagging NN Training"):
            vecs, scores, labels = batch
            vecs = vecs.to(device)
            scores = scores.to(device)
            labels = labels.to(device)

            s, loss = model(vecs, scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1
            total_loss += loss.item()

        print("train loss: {}".format(total_loss / total_steps))

        rmse, preds = evaluate()

        print("evaluate rmse: {}".format(rmse))
