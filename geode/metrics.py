# metrics.py

import numpy as np
from numpy import asarray, ndarray


def convert_and_flatten(y_true: ndarray,
                        y_pred: ndarray,
                        pos_label: int) -> tuple:

    """Converts input arrays to binary labels, and flattens them.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        A tuple containing the converted and flattened arrays.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    y_true = np.where(y_true.flatten() == pos_label, 1, 0)
    y_pred = np.where(y_pred.flatten() == pos_label, 1, 0)

    return y_true, y_pred


def true_positives(y_true: ndarray,
                   y_pred: ndarray,
                   pos_label: int) -> int:

    """Computes the true positives between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The number of true positives between the arrays.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return int(np.sum(y_true * y_pred))


def false_positives(y_true: ndarray,
                    y_pred: ndarray,
                    pos_label: int) -> int:

    """Computes the false positives between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The number of false positives between the arrays.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return int(np.sum(np.where(y_pred - y_true == 1, 1, 0)))


def false_negatives(y_true: ndarray,
                    y_pred: ndarray,
                    pos_label: int) -> int:

    """Computes the false negatives between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The number of false negatives between the arrays.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    y_true, y_pred = convert_and_flatten(y_true, y_pred, pos_label)

    return int(np.sum(np.where(y_true - y_pred == 1, 1, 0)))


def f1(y_true: ndarray,
       y_pred: ndarray,
       pos_label: int) -> float:

    """Computes the F1 (aka, dice) score between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The F1 score.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + 0.5 * (fp + fn))


def jaccard(y_true: ndarray,
            y_pred: ndarray,
            pos_label: int) -> float:

    """Computes the Jaccard (aka, intersection-over-union) score between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The Jaccard score.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)

    return tp / (tp + fp + fn)


def precision(y_true: ndarray,
              y_pred: ndarray,
              pos_label: int) -> float:

    """Computes the precision score between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The precision score.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

    tp = true_positives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)

    return tp / (tp + fp)


def recall(y_true: ndarray,
           y_pred: ndarray,
           pos_label: int) -> float:

    """Computes the recall score between the ground-truth and predicted array.

    Args:
        y_true: an array of true labels;
        y_pred: an array of predicted labels;
        pos_label: the label in y_true and y_pred to change to 1.

    Returns:
        The recall score.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    pos_label = int(pos_label)

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

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)

    # check whether the two arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise Exception("The provided arrays do not have the same shape.")

    # compute accuracy
    acc = np.sum(np.where(y_true == y_pred, 1, 0)) / np.prod(y_true.shape)

    return acc
