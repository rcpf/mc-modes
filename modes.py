import numpy as np
import pandas as pd

from copy import copy, deepcopy
from sklearn.metrics import pairwise_distances

from des_thr import DESthr
from clustering import execute_nbclust
from utils import split_by_label


class Modes(DESthr):
    def __init__(self, occ, max_clusters=10):
        super().__init__(occ)
        self.max_clusters = max_clusters

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.labels = sorted(set(y))

        # Split training data by class
        tr = split_by_label(self.X_train, self.y_train, self.labels)
        for label in self.labels:
            occ_l = copy(self.occ)
            ocdes = OcDES(occ_l)
            ocdes.fit(tr[label])
            self.occs[label] = ocdes


class OcDES:
    def __init__(self, classifier, max_clusters=10):
        self.max_clusters = max_clusters
        self.occ = classifier
        self.map_cluster_occ = dict()
        self.map_centroids = dict()
        self.classifiers = dict()
        self.data = None
        self.n_partitions = None

    def fit(self, data):
        self.data = data
        self.n_partitions = self._evaluate_partitions()
        n_temp = deepcopy(self.n_partitions)
        for cluster_number in n_temp:
            clusters_labels = list(set(self.n_partitions[cluster_number]))
            X_cluster = split_by_label(self.data, self.n_partitions[cluster_number], clusters_labels)
            self.map_centroids[cluster_number] = compute_centroids(X_cluster)
            self.classifiers[cluster_number] = dict()
            for cluster_label in clusters_labels:
                # Aqui considero que instâncias repetidas como uma única instância (mst_dd dá erro caso haja apenas
                # 1 instância de treino ou instâncias iguais)
                # Uma consequencia disso eh que, como considero clusters com um único exemplo como ruido,
                # um cluster que tenha 200 instancias repetidas, será removido por ser considerado ruido
                X = np.unique(X_cluster[cluster_label], axis=0)
                if len(X) > 1:
                    occ = copy(self.occ)
                    occ.fit(X)
                    self.classifiers[cluster_number][cluster_label] = occ

    def score_samples(self, X_test):
        decisions = pd.DataFrame(columns=self.n_partitions)
        # Clustering
        for cluster_number in self.n_partitions:
            if self.map_centroids[cluster_number] == 1:
                clust_pred = np.zeros(len(X_test))
                pred = np.full((len(X_test),), np.nan, dtype=float)
                occ = self.classifiers[cluster_number][1]
                Xx = X_test[clust_pred == 1]
                if len(Xx) > 0:
                    pr = occ.score_samples(Xx)
                    pred[clust_pred == 1] = np.array(pr).reshape((len(Xx),))    # TODO - essa linha é desnecessária
                decisions[cluster_number] = pred
            else:
                clust_pred = nearest_centroid(self.map_centroids[cluster_number], X_test)
                pred = np.full((len(X_test),), np.nan, dtype=float)
                for cluster_label in list(set(self.n_partitions[cluster_number])):
                    if cluster_label not in self.classifiers[cluster_number]:  # case of discarded cluster in the training phase (contains only one example)
                        continue
                    occ = self.classifiers[cluster_number][cluster_label]
                    Xx = X_test[clust_pred == cluster_label]
                    if len(Xx) > 0:
                        pr = occ.score_samples(Xx)
                        pred[clust_pred == cluster_label] = np.array(pr).reshape((len(Xx),))
                decisions[cluster_number] = pred
        cls_decision = decisions.mean(axis=1, skipna=True)  # default
        return cls_decision

    def _evaluate_partitions(self):
        self._check_samples_size()
        if self.max_clusters > 6:
            partitions = execute_nbclust(self.data, self.max_clusters)
        else:   # For small sample sizes, all the examples are clustered toghether
            partitions = {1: np.array([1] * len(self.data))}
        return partitions

    def _check_samples_size(self):
        # The number of clusters can't surpass the number of examples in the class
        len_data = len(np.unique(self.data, axis=0))
        if len_data <= self.max_clusters:
            self.max_clusters = len_data - 1


def compute_centroids(X_cluster):
    """
    Given the data for each cluster and the cluster labels, return the centroids for each cluster
    :param X_cluster: dictionary containing pairs of cluster labels as keys and cluster data as values
    :return: the centroid for each cluster
    """
    centroids = {}
    for cluster_label in X_cluster:
        centroids[cluster_label] = (np.mean(X_cluster[cluster_label], axis=0))
    return centroids


def nearest_centroid(centroids, X):
    """
    Assign each example in X for the nearest cluster, based on the euclidean distance to the clusters centroids
    :param centroids: the clusters centroids
    :param X: the data to be labeled
    :return: an array with the cluster label for each example in X
    """
    Y = np.array([centroids[c] for c in centroids])
    nearest = np.argmin(pairwise_distances(Y, X, metric='euclidean').T, axis=1)
    return np.array(list(centroids.keys()))[nearest]
