from copy import copy

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from utils import split_by_label


class OccDecomposition(BaseEstimator):
    def __init__(self, occ):
        """

        :param occ: An instance of an one-class classifier that implements fit(X) and score_samples(X).
        """
        self.occ = occ
        self.X_train = None
        self.y_train = None
        self.labels = None
        self.scores = None
        self.occs = {}

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.labels = list(set(y_train))

        X_train_by_label = split_by_label(self.X_train, self.y_train, self.labels)
        for label in self.labels:
            if len(X_train_by_label[label]) > 1:    # TODO - SVDD cant't fit the model with a single example
                local_occ = copy(self.occ)
                local_occ.fit(X_train_by_label[label])
                self.occs[label] = local_occ

    def predict(self, X_test):
        self.scores = pd.DataFrame(index=list(range(len(X_test))))
        for label in self.labels:
            occ = self.occs[label]
            self.scores[label] = occ.score_samples(X_test)
        self.scores = self.scores.astype('float64')
        # Here we use max aggregation for sake of simplicity. Feel free to use other aggregation techniques ;)
        return np.array(self.scores.idxmax(axis=1, skipna=True))

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)
