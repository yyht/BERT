import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import

BATCH_NORM_EPSILON = 1e-5


class BatchNormalization(tf.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation on TPU.
  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica (TPU core).
  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  """

  def __init__(self, fused=False, **kwargs):
    """Builds the batch normalization layer.
    Arguments:
      fused: If `False`, use the system recommended implementation. Only support
        `False` in the current implementation.
      **kwargs: input augments that are forwarded to
        tf.layers.BatchNormalization.
    """
    if fused in (True, None):
      raise ValueError('The TPU version of BatchNormalization does not support '
                       'fused=True.')
    super(BatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    return tf.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(BatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards and num_shards > 1:
      # Each group has multiple replicas: here we compute group mean/variance by
      # aggregating per-replica mean/variance.
      group_mean = self._cross_replica_average(shard_mean)
      group_variance = self._cross_replica_average(shard_variance)

      # Group variance needs to also include the difference between shard_mean
      # and group_mean.
      mean_distance = tf.square(group_mean - shard_mean)
      group_variance += self._cross_replica_average(mean_distance)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def batch_norm_relu(FLAGS, inputs, is_training, relu=True, init_zero=False,
                    center=True, scale=True, data_format='channels_last'):
  """Performs a batch normalization followed by a ReLU.
  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    center: `bool` whether to add learnable bias factor.
    scale: `bool` whether to add learnable scaling factor.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  if FLAGS.global_bn:
    bn_foo = BatchNormalization(
        axis=axis,
        momentum=FLAGS.batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        fused=False,
        gamma_initializer=gamma_initializer)
    inputs = bn_foo(inputs, training=is_training)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=FLAGS.batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs