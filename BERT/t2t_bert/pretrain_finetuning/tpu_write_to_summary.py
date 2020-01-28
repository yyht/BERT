import tensorflow as tf
import numpy as np
import tensorflow.compat.v2 as tf2

def host_call_fn(model_dir, gs, log_dict):
    """Training host call. Creates scalar summaries for training metrics.
    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the
    model to the `metric_fn`, provide as part of the `host_call`. See
    https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
    for more information.
    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `host_call`.
    Args:
      gs: `Tensor with shape `[batch]` for the global_step
      loss: `Tensor` with shape `[batch]` for the training loss.
      lr: `Tensor` with shape `[batch]` for the learning_rate.
      ce: `Tensor` with shape `[batch]` for the current_epoch.
    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = gs[0]
    # Host call fns are executed params['iterations_per_loop'] times after
    # one TPU loop is finished, setting max_queue value to the same as
    # number of iterations will make the summary writer only flush the data
    # to storage once per loop.
    with tf2.summary.create_file_writer(
        model_dir,
        max_queue=1000).as_default():
		with tf2.summary.record_if(True):
			for key in log_dict:
				tf2.summary.scalar(key, log_dict[key][0], step=gs)
      return tf.summary.all_v2_summary_ops()

  # To log the loss, current learning rate, and epoch for Tensorboard, the
  # summary op needs to be run on the host CPU via host_call. host_call
  # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
  # dimension. These Tensors are implicitly concatenated to
  # [params['batch_size']].
  gs_t = tf.reshape(global_step, [1])
  loss_t = tf.reshape(loss, [1])
  lr_t = tf.reshape(learning_rate, [1])
  ce_t = tf.reshape(current_epoch, [1])

  host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])