# losses.py

from numpy import asarray, ndarray
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_loss(y_true: ndarray,
              y_pred: ndarray,
              smooth: float = 100.) -> float:
    """Computes the dice-loss between one-hot encoded arrays, then returns a score between [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width, n_classes);
        y_pred: tensor of model predictions of size (batch, height, width, n_classes);
        smooth: a value to avoid division by zero (among other things).

    Returns:
        The dice-loss score.

    Raises:
        ValueError: if smooth is less than zero.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    smooth = float(smooth)

    # check for the correct float range
    if smooth < 0:
        raise ValueError("Argument smooth must be greater than or equal to zero.")

    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1 - dice


def iou_loss(y_true: ndarray,
             y_pred: ndarray,
             smooth: float = 100.) -> float:
    """Computes the IoU-loss between one-hot encoded arrays, then return a loss score between [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width);
        y_pred: tensor of model predictions of size (batch, height, width);
        smooth: a value to avoid division by zero (among other things).

    Returns:
        The iou-loss score.

    Raises:
        ValueError: if smooth is less than zero.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    smooth = float(smooth)

    # check for the correct float range
    if smooth < 0:
        raise ValueError("Argument smooth must be greater than or equal to zero.")

    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)

    intersection = K.sum(K.flatten(y_true * y_pred))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    iou = intersection / union

    return 1 - iou


def log_iou_loss(y_true: ndarray,
                 y_pred: ndarray,
                 smooth: float = 100.) -> float:
    """Computes a variation of the -log(IoU) loss introduced in 'Unitbox': An Advanced Object Detection Network, with
    a smoothing parameter to (among other things) avoid division by zero.

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width);
        y_pred: tensor of model predictions of size (batch, height, width);
        smooth: a value to avoid division by zero (among other things).

    Returns:
        The log-iou-loss score.

    Raises:
        ValueError: if smooth is less than zero.
    """

    # attempt to coerce arguments to the correct type
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)
    smooth = float(smooth)

    # check for the correct float range
    if smooth < 0:
        raise ValueError("Argument smooth must be greater than or equal to zero.")

    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)

    intersection = K.sum(K.flatten(y_true * y_pred)) + smooth
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + smooth

    final = - K.log(intersection / union)

    return final
