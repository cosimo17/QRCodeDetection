import tensorflow as tf


def yolo_loss(y_true, y_pred):
    """
    :param y_true: [n, gridw, gridh, anchor_per_grid, channel]
    :param y_pred: [n, gridw, gridh, anchor_per_grid, channel]
    :return: loss
    """
    pred_scores = tf.math.sigmoid(y_pred[..., 0])
    pred_cls = tf.math.softmax(y_pred[..., 1:3], axis=-1)
    epsilon = 0.0001
    pred_cls = tf.clip_by_value(pred_cls, epsilon, 1 - epsilon)
    pred_xy = tf.math.tanh(y_pred[..., 3:5])
    pred_wh = tf.math.tanh(y_pred[..., 5:])

    true_scores = y_true[..., 0]
    true_cls = y_true[..., 1:3]
    true_xy = y_true[..., 3:5]
    true_wh = y_true[..., 5:]

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    score_loss = bce(true_scores, pred_scores)

    cls_mask = true_scores + 0.005
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    cls_loss = cce(true_cls, pred_cls) * cls_mask
    cls_loss = tf.math.reduce_mean(cls_loss)

    se = lambda x, y: tf.reduce_sum(tf.math.square(x - y), axis=-1)
    xy_loss = se(true_xy, pred_xy) * true_scores
    wh_loss = se(true_wh, pred_wh) * true_scores
    bbox_loss = xy_loss + wh_loss
    bbox_loss = tf.math.reduce_mean(bbox_loss)

    loss = score_loss + 2 * cls_loss + 5 * bbox_loss
    loss *= 32

    return loss
