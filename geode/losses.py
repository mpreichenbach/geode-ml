# losses.py

import tensorflow as tf
from tensorflow.keras import backend as K


def mean_iou_loss(y_true, y_pred) -> float:
    """Computes the IoU for each class, averages them, then returns a loss score between [0, 1].

    Args:
        y_true: tensor of ground-truth values of size(batch, height, width, n_classes);
        y_pred: tensor of model predictions of size (batch, height, width, n_classes).

    Returns:
        The mean-IoU loss score."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = y_true * y_pred
    union = y_true + y_pred - intersection

    miou = K.sum(K.flatten(K.mean(intersection, axis=-1))) / (K.sum(K.flatten(K.mean(union, axis=-1))) + 1)

    return miou

def iou_loss(y_true, y_pred):
    """Computes the IoU for a single target class, then return a loss score between [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width);
        y_pred: tensor of model predictions of size (batch, height, width)."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(K.flatten(y_true * y_pred))
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + 1

    final = 1 - intersection / union

    return final

def log_iou_loss(y_true, y_pred):
    """Compute a variation of the -log(IoU) loss introduced in 'Unitbox': An Advanced Object Detection Network. This
    version includes the 'smooth' parameter to ensure no division by zero.

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width), so not one-hot encoded;
        y_pred: tensor of model predictions of size (batch, height, width, 2), so one-hot encoded."""

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(K.flatten(y_true * y_pred)) + 1
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + 1

    final = - K.log(intersection / union)

    return final
