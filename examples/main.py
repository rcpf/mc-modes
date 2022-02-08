import numpy as np

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

import modes as md

from occ_models.base_svdd import BaseSVDD


X, y = load_iris(return_X_y=True)
y = y.astype(str)

pipe_occ_decomposition = make_pipeline(
    MinMaxScaler(),
    md.Modes(BaseSVDD()),
)

pipe_modes = make_pipeline(
    MinMaxScaler(),
    md.Modes(OneClassSVM()),
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipe_occ_decomposition,  X, y, cv=skf)
print('Modes(SVDD): {0:.2f} ({1:.2f})'.format(np.mean(scores)*100, np.std(scores)*100))

scores = cross_val_score(pipe_modes,  X, y, cv=skf)
print('Modes(OcSVM): {0:.2f} ({1:.2f})'.format(np.mean(scores)*100, np.std(scores)*100))
