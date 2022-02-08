import numpy as np
import pandas as pd

from collections import Counter
from occ_decomposition import OccDecomposition
from sklearn.neighbors import NearestNeighbors


class DESthr(OccDecomposition):

    def __init__(self, occ):
        super().__init__(occ)
        self.neighborhood = None

    def classes_in_neighborhood(self, X_test):
        k = 3 * len(self.labels)

        neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.X_train)
        _, indices = neighbors.kneighbors(X_test)
        return [[ct for ct in c if c[ct] >= 0.1 * k] for c in [Counter(self.y_train[row]) for row in indices]]

    def predict(self, X_test):
        self.neighborhood = self.classes_in_neighborhood(X_test)
        # if the neighborhood contains only one class label, 1-NN is applied
        predictions = np.array([j[0] if len(j) == 1 else None for j in np.array(self.neighborhood, dtype=object)])
        idx_n_resp = list(np.where(predictions == None)[0])  # Only examples with more than one class label in the neighborhood

        self.scores = pd.DataFrame(index=list(range(len(X_test))), columns=self.labels, dtype='float64')
        for label in self.labels:
            idx = list(np.where([label in nb and len(nb) > 1 for nb in np.array(self.neighborhood, dtype=object)])[0])
            if idx:
                occ = self.occs[label]
                label_scores = self.scores[label].values
                label_scores[idx] = np.ravel(occ.score_samples(X_test[idx]))
                self.scores[label] = label_scores
        predictions[idx_n_resp] = np.array(self.scores.loc[idx_n_resp].idxmax(axis=1, skipna=True), dtype='<U64')
        return predictions
