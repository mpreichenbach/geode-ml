# losses.py

from tensorflow.keras import backend as K


def dice_loss(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1 - dice


def iou_loss(y_true, y_pred):
    """Computes the IoU for a single target class, then return a loss score between [0, 1].

    Args:
        y_true: tensor of ground-truth values of size (batch, height, width);
        y_pred: tensor of model predictions of size (batch, height, width)."""
    # use tf.math.reduce_sum
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

    intersection = K.sum(K.flatten(y_true * y_pred)) + 1
    union = K.sum(K.flatten(y_true + y_pred - y_true * y_pred)) + 1

    final = - K.log(intersection / union)

    return final
