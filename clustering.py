import numpy as np
import pandas as pd

from rpy2.robjects import r, pandas2ri, default_converter, conversion


def execute_nbclust(data, max_clusters):

    """
    :param data: Training data
    :param max_clusters: Maximum number of clusters to be evaluated
    :return: A dict containing the number of clusters as key and a list of cluster labels for each training example as value
    """

    indices = [
        'ball',
        'ch',
        'cindex',
        'db',
        'dunn',
        'gap',
        'hartigan',
        'kl',
        'mcclain',
        'ptbiserial',
        'sdbw',
        'sdindex',
        'silhouette',
    ]

    r2py = r
    # Suppress warnings
    r2py['options'](warn=-1)
    r2py.source('BestPartitions.R')
    pandas2ri.activate()

    with conversion.localconverter(default_converter + pandas2ri.converter):
        r_dataframe = conversion.py2rpy((pd.DataFrame(data)))
    r_return = r2py.computePartitions(r_dataframe, indices, max_clusters)
    best_partitions = dict(zip(np.array(r_return.names, dtype='int'), map(np.array, np.array(r_return))))
    return best_partitions
