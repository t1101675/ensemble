from sklearn import tree
from sklearn.svm import SVC, LinearSVC
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score

class DTree():
    def __init__(self):
        self.cls = tree.DecisionTreeClassifier(max_depth=30)

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
    def __init__(self):
        # self.cls = SVC(kernel='linear', verbose=False)
        self.cls = LinearSVC()

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
