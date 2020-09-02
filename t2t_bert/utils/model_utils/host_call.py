
import tensorflow as tf
"""
https://gitlab.aiacademy.tw/at081045/at081-group5/blob/master/FasterRCNN_BBB/models/official/utils/accelerator/tpu.py
"""


def construct_scalar_host_call(metric_dict, model_dir, prefix=""):
	"""Construct a host call to log scalars when training on TPU.

	Args:
		metric_dict: A dict of the tensors to be logged.
		model_dir: The location to write the summary.
		prefix: The prefix (if any) to prepend to the metric names.

	Returns:
		A tuple of (function, args_to_be_passed_to_said_function)
	"""
	# type: (dict, str) -> (function, list)
	metric_names = list(metric_dict.keys())

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
	other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

	return host_call_fn, [global_step_tensor] + other_tensors


