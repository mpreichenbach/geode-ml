# metrics.py

import numpy as np
from numpy import ndarray

def convert_and_flatten(y_true, y_pred, pos_label):
    y_true = np.where(y_true.flatten() == pos_label, 1, 0)
    y_pred = np.where(y_pred.flatten() == pos_label, 1, 0)

    return y_true, y_pred

def true_positives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(y_true * y_pred)

def false_positives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(np.where(y_pred - y_true == 1, 1, 0))

def false_negatives(y_true, y_pred, pos_label):
    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return np.sum(np.where(y_true - y_pred == 1, 1, 0))

# these functions are way faster than their sklearn counterparts

def f1(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + 0.5 * (fp + fn))

def jaccard(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + fp + fn)

def precision(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)

    return tp / (tp + fp)

def recall(y_true, y_pred, pos_label):
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + fn)

def total_accuracy(y_true: ndarray,
                   y_pred: ndarray) -> float:

    """Computes the proportion of correctly predicted labels.

    Args:
        y_true: the array of true labels;
        y_pred: the array of predicted labels.

    Returns:
        The proportion of correctly predicted labels.

    Raises:
        Exception: if the two provided arrays do not have the same shape."""

    # check whether the two arrays have the same shape
    if y_true.shape == y_pred.shape:
        pass
    else:
        raise Exception("The provided arrays do not have the same shape.")

    # compute accuracy
    acc = np.sum(np.where(y_true == y_pred, 1, 0)) / np.prod(y_true.shape)

    return acc
