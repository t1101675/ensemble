from sklearn import tree
from sklearn.svm import SVC, LinearSVC
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import Adam
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score

class DTree():
    def __init__(self, max_depth):
        self.cls = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced")

    def train(self, data, labels, sample_weight=None):
        self.cls.fit(data, labels, sample_weight=sample_weight)
        pred = self.cls.predict(data)
        acc = accuracy_score(labels, pred)
        rmse = mean_squared_error(labels, pred)
        return acc, rmse, pred

    def eval(self, data, labels):
        pred = self.cls.predict(data)
        acc = accuracy_score(labels, pred)
        rmse = mean_squared_error(labels, pred) ** 0.5
        return acc, rmse, pred

    def predict(self, data):
        pred = self.cls.predict(data)
        return pred

    def save(self, path):
        joblib.dump(self.cls, path)

    def load(self, path):
        self.cls = joblib.load(path)

class SVM():
    def __init__(self, tol=1e-4, C=1):
        # self.cls = SVC(kernel='linear', verbose=False)
        self.cls = LinearSVC(tol=tol, C=C, class_weight="balanced")

    def train(self, data, labels, sample_weight=None):
        # self.cls.fit(data, labels, sample_weight=sample_weight * len(labels))
        if sample_weight is not None:
            self.cls.fit(data, labels, sample_weight=sample_weight * len(labels))
        else:
            self.cls.fit(data, labels)
            
        pred = self.cls.predict(data)
        acc = accuracy_score(labels, pred)
        rmse = mean_squared_error(labels, pred) ** 0.5
        return acc, rmse, pred
    
    def eval(self, data, labels):
        pred = self.cls.predict(data)
        acc = accuracy_score(labels, pred)
        rmse = mean_squared_error(labels, pred) ** 0.5
        return acc, rmse, pred

    def predict(self, data):
        pred = self.cls.predict(data)
        return pred

    def save(self, path):
        joblib.dump(self.cls, path)

    def load(self, path):
        self.cls = joblib.load(path)

class RNN(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, device, n_class=5):
        super().__init__()
        self.rnn = nn.RNN(n_embd, n_hidden, batch_first=True)
        self.cls = nn.Linear(n_hidden, n_class)
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.a = torch.tensor([0.0, 1, 2, 3, 4], requires_grad=False).to(device)

    def forward(self, x, lengths, labels=None):
        x = self.wte(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, hn = self.rnn(x)
        x, ls = pad_packed_sequence(x, batch_first=True)
        logits = self.cls(x.mean(dim=1))
        logits = nn.functional.softmax(logits, dim=1).matmul(self.a)
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class RNNDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = sorted(data, key=lambda x: len(x), reverse=True)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype=torch.long), self.labels[item], len(self.data[item])

def rnn_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    data = pad_sequence(data, batch_first=True)
    return data, torch.tensor(labels, dtype=torch.float), lengths


class NN():
    def __init__(self, vocab_size, n_embd, n_hidden, train_batch_size, valid_batch_size, device, epoch):
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.model = RNN(vocab_size, n_embd, n_hidden, device).to(device)
        self.epoch = epoch
        self.device = torch.device(device)

    def train(self, data, labels):
        print("NN start training")
        dataset = RNNDataset(data, labels)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=self.train_batch_size, collate_fn=rnn_collate, sampler=sampler)
        optimizer = Adam(self.model.parameters(), lr=5e-4)
        self.model.train()
        total_steps = 0
        total_loss = 0
        preds = []
        for e in tqdm(range(self.epoch), desc="NN Training"):
            for batch in tqdm(data_loader):
                input_ids, label, lengths = batch
                input_ids = input_ids.to(self.device)
                label = label.to(self.device)
                loss, pred = self.model(input_ids, lengths, label)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                total_steps += 1
                total_loss += loss.item()

                preds.extend(pred.detach().to(torch.device("cpu")).numpy().tolist())
            print("loss: {}".format(total_loss / total_steps))

        rmse = (total_loss / total_steps) ** 0.5
        acc = 0
        return acc, rmse, preds
        
    def eval(self, data, labels):
        dataset = RNNDataset(data, labels)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=self.valid_batch_size, collate_fn=rnn_collate, sampler=sampler)
        self.model.eval()
        total_steps = 0
        total_loss = 0
        preds = []
        for batch in tqdm(data_loader, desc="NN Evaling"):
            input_ids, label, lengths = batch
            input_ids = input_ids.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                loss, pred = self.model(input_ids, lengths, label)
                total_steps += 1
                total_loss += loss.item()

                preds.extend(pred.to(torch.device("cpu")).numpy().tolist())
        
        rmse = (total_loss / total_steps) ** 0.5
        acc = 0
        return rmse, preds


    def predict(self, data):
        dataset = RNNDataset(data, [0] * len(data))
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=self.valid_batch_size, collate_fn=rnn_collate, sampler=sampler)
        self.model.eval()
        preds = []
        for batch in tqdm(data_loader, desc="NN Evaling"):
            input_ids, label, lengths = batch
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                pred = self.model(input_ids, lengths)
                preds.extend(pred.to(torch.device("cpu")).numpy().tolist())
        
        return preds
