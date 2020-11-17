
try:
	import tensorflow.compat.v2 as tf2
except:
	tf2 = None

import tensorflow as tf

def construct_scalar_host_call_v1(monitor_dict,
    							model_dir, prefix=""):
  """Construct a host call to log scalars when training on TPU.

  Args:
    monitor_dict: A dict of the tensors to be logged.
    model_dir: The location to write the summary.
    prefix: The prefix (if any) to prepend to the metric names.

  Returns:
    A tuple of (function, args_to_be_passed_to_said_function)
  """
  # type: (dict, str) -> (function, list)
  metric_names = list(monitor_dict.keys())

  def host_call_fn(global_step, *args):
    """Training host call. Creates scalar summaries for training metrics.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the
    model to the `metric_fn`, provide as part of the `host_call`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `host_call`.

    Args:
      global_step: `Tensor with shape `[batch]` for the global_step
      *args: Remaining tensors to log.

    Returns:
      List of summary ops to run on the CPU host.
    """
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          tf.contrib.summary.scalar(prefix + name, args[i][0], step=step)

        return tf.contrib.summary.all_summary_ops()

  # To log the current learning rate, and gradient norm for Tensorboard, the
  # summary op needs to be run on the host CPU via host_call. host_call
  # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
  # dimension. These Tensors are implicitly concatenated to
  # [params['batch_size']].
  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(monitor_dict[key], [1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def construct_scalar_host_call_v2(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
  """Construct host call for scalar."""

  # Only consider scalar
  metric_names = []
  for k, v in sorted(monitor_dict.items(), key=lambda x: x[0]):
    if v.shape.ndims == 0:
      metric_names.append(k)
      tf.logging.info("Host call receives %s: %s", k, v.shape)
      monitor_dict[k] = tf.reshape(v, [1])
    else:
      tf.logging.info("Host call ignores %s: %s", k, v.shape)

  def host_call_fn(global_step, *args):
    """Actual host call function."""
    step = global_step[0]
    with tf2.summary.create_file_writer(
        model_dir, filename_suffix=".host_call").as_default():
      with tf2.summary.record_if(lambda: tf.equal(step % 1000, 0)):
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          tf2.summary.scalar(prefix + name, scalar, step=step)

        return tf.summary.all_v2_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [monitor_dict[key] for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors