import tensorflow as tf
import numpy as np

EPSILON = 1e-30

def focal_loss_binary_v2(config, logits, labels):
    """
    alpha = 0.5
    gamma = 2.0
    """
    alpha = config.get("alpha", 0.5)
    gamma = config.get("gamma", 2.0)

    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

    predictions = tf.nn.softmax(logits)
    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)

    labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

    postive_loss = labels * tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)* alpha
    negative_loss = (1-labels)*tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma) * (1 - alpha)

    losses = -postive_loss - negative_loss
    return tf.reduce_mean(losses), predictions

def focal_loss_multi_v1(config, logits, labels):
    gamma = config.get("gamma", 2.0)

    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

    predictions = tf.exp(tf.nn.log_softmax(logits))

    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)

    labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

    losses =  tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)

    return -losses

def weighted_loss_ratio(config, losses, labels, ratio_weight):
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    # appear_times = tf.reshape(appear_times, [-1, 1])

    weighted_loss = losses * ratio_weight
    weighted_loss = weighted_loss / tf.cast((EPSILON+appear_times), tf.float32)

    return weighted_loss

def magrin_loss(config, logits, labels):
    pass


