import tensorflow as tf
import numpy as np

def bf16_decorator(func):
	"""A wrapper function for bfloat16 scope."""
	@functools.wraps(func)
	def wrapped_func(*args, **kwargs):
		if FLAGS.use_bfloat16:
			with tf.tpu.bfloat16_scope():
				return func(*args, **kwargs)
		else:
			with tf.variable_scope(""):
				return func(*args, **kwargs)

	return wrapped_func