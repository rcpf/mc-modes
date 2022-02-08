
def split_by_label(data, targets, labels):
    """
    Given a data matrix and the targets for the data, the method splits the data by class
    :return: a dict with the data separeted by label, each label is a key of the dict
    """
    d = dict()
    for cl in labels:
        d[cl] = data[targets == cl]
    return d
