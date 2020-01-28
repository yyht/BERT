import numpy as np

import tensorflow as tf

class MovingAverageOptimizer(tf.train.Optimizer):
	def __init__(self, opt, average_decay=0.9999, num_updates=None,
			   sequential_update=True):
		"""Construct a new MovingAverageOptimizer.
		"""
		self._optimizer = opt
		self._ema = ExponentialMovingAverage(
			average_decay, num_updates=num_updates, zero_debias=True)
		self._swapped_variable_name_map = None
		self._sequential_update = sequential_update

	def compute_gradients(self, *args, **kwargs):
		return self._optimizer.compute_gradients(*args, **kwargs)

	def apply_gradients(self, grads_and_vars, global_step=None, name=None):
		train_op = self._optimizer.apply_gradients(
			grads_and_vars, global_step=global_step, name=name)
		var_list = [x[1] for x in grads_and_vars if x[0] is not None]
		self._swapped_variable_name_map = {}
		if self._sequential_update:
			with tf.control_dependencies([train_op]):
				ma_op = self._ema.apply(var_list)
		else:
			ma_op = self._ema.apply(var_list)

		for v in var_list:
			v_avg = self._ema.average(v)
			self._swapped_variable_name_map[v.op.name] = v_avg.op.name
			self._swapped_variable_name_map[v_avg.op.name] = v.op.name
		return tf.group(train_op, ma_op, name='train_with_avg')

	def swapping_saver(self, var_list=None, name='swapping_saver', **kwargs):
		"""Create a saver swapping moving averages and variables.
		"""
		if self._swapped_variable_name_map is None:
			raise RuntimeError('Must call apply_gradients or minimize before '
							 'creating the swapping_saver')
		if var_list is None:
			var_list = tf.global_variables()
		v_name_to_tensor = {v.op.name: v for v in var_list}

		# Now swap variables and moving averages
		swapped_var_list = {}
		for v_name, v in v_name_to_tensor.items():
			swapped_v_name = self._swapped_variable_name_map.get(v_name, None)
			v_to_save = v
			if swapped_v_name is not None:
				if swapped_v_name in v_name_to_tensor:
					v = v_name_to_tensor[swapped_v_name]
				else:
					raise ValueError(
						('Variable to swap %s is not part of variables to save. '
						'This breaks MovingAverageOptimizer.') % swapped_v_name)
			swapped_var_list[v_name] = v

		# Build the swapping saver.
		return tf.train.Saver(swapped_var_list, name=name, **kwargs)