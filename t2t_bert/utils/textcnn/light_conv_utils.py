import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers
from utils.bert import bert_utils
import tensorflow.contrib.layers as layers
from utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from utils.qanet import qanet_layers
from utils.qanet.qanet_layers import highway
from utils.dsmm.tf_common.nn_module import encode, attend, mlp_layer
from utils.bert import bert_utils
from utils.esim import esim_utils

initializer = tf.glorot_uniform_initializer()
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
															 mode='FAN_AVG',
															 uniform=True,
															 dtype=tf.float32)

def layer_norm_compute_python(x, epsilon, scale, bias):
	"""Layer norm raw computation."""
	mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
	variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
	norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
	return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
	"""Layer normalize the tensor x, averaging over the last dimension."""
	if filters is None:
		filters = x.get_shape()[-1]
	with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
		scale = tf.get_variable(
			"layer_norm_scale", [filters], initializer=tf.ones_initializer())
		bias = tf.get_variable(
			"layer_norm_bias", [filters], initializer=tf.zeros_initializer())
		result = layer_norm_compute_python(x, epsilon, scale, bias)
		return result

def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
									scope = "depthwise_separable_convolution",
									padding='SAME',
									bias = True, 
									is_training = True, 
									reuse = None,
									activation = tf.nn.relu,
									kernel_initializer = None,
									strides = 1,
									dilation_rate = 1):
	outputs = tf.expand_dims(inputs, 2) # batch, seq, 1, dim
	shapes = bert_utils.get_shape_list(outputs, expected_rank=[4])
	with tf.variable_scope(scope, reuse = reuse):
		depthwise_filter = tf.get_variable("depthwise_filter",
										(kernel_size[0], kernel_size[1], shapes[-1], 1),
										dtype = tf.float32,
										initializer = kernel_initializer)
		pointwise_filter = tf.get_variable("pointwise_filter",
										(1, 1, shapes[-1], num_filters),
										dtype = tf.float32,
										initializer = kernel_initializer)

		outputs = tf.nn.separable_conv2d(outputs,
										depthwise_filter,
										pointwise_filter,
										strides = (1, strides, 1, 1),
										padding = padding,
										rate = (dilation_rate, 1))
		if bias:
			b = tf.get_variable("bias",
					outputs.shape[-1],
					# regularizer=regularizer,
					initializer = tf.zeros_initializer())
			outputs += b
		if activation:
			outputs = activation(outputs)

		return tf.squeeze(outputs,2)

def gated_conv1d_op(inputs, 
				filters=8, 
				kernel_size=3, 
				padding="same", 
				activation=None, 
				strides=1, 
				reuse=False, 
				name="", 
				kernel_initializer=None,
				dilation_rate=1,
				is_training=True):
	conv_linear = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_linear",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv_gated = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_gated",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = tf.nn.sigmoid,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv = conv_linear * conv_gated
	return conv

def residual_gated_conv1d_op(inputs, 
							residual_inputs,
							filters=8, kernel_size=3, 
							padding='same',
							activation=None, 
							strides=1, 
							reuse=False, 
							dilation_rate=1,
							name="",
							kernel_initializer=None, 
							is_training=False):
	conv_linear = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_linear",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	conv_gated = depthwise_separable_convolution(
		inputs = inputs,
		kernel_size = [kernel_size, 1],
		num_filters = filters,
		scope = name+"_gated",
		padding=padding,
		bias = True, 
		is_training = True, 
		reuse = reuse,
		activation = None,
		kernel_initializer = kernel_initializer,
		strides = strides,
		dilation_rate = dilation_rate,
		)
	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0
	conv_gated = tf.nn.sigmoid(tf.nn.dropout(conv_gated, 1-dropout_rate))
	conv = residual_inputs * (1. - conv_gated) + conv_linear * conv_gated
	return conv

def dgcnn(x, input_mask,
			num_layers=2, 
			dilation_rates=[1,2],
			strides=[1,1],
			num_filters=[64,64],
			kernel_sizes=[3,3], 
			is_training=False,
			scope_name="textcnn", 
			reuse=False, 
			activation=tf.nn.relu,
			is_casual=False,
			padding='SAME'
			):

	# input_mask: batch_size, seq

	# initializer = tf.glorot_uniform_initializer()
	initializer = create_initializer(initializer_range=0.02)

	input_mask = tf.cast(input_mask, dtype=tf.float32)
	input_mask = tf.expand_dims(input_mask, axis=-1)

	if is_casual:
		left_pad = dilation_rates[0] * (kernel_sizes[0] - 1)
		inputs = tf.pad(x, [[0, 0, ], [left_pad, 0], [0, 0]])
		padding = 'VALID'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = x
		padding = 'SAME'

	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0

	with tf.variable_scope(scope_name, reuse=reuse):
		inputs = gated_conv1d_op(inputs,
						filters=num_filters[0],
						kernel_size=kernel_sizes[0],
						padding=padding,
						activation=None,
						strides=1,
						reuse=reuse, 
						dilation_rate=1,
						name="gated_conv",
						kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1),
						is_training=is_training)
		if padding == 'SAME':
			inputs *= input_mask
		residual_inputs = inputs

	for (dilation_rate, 
		layer, 
		kernel_size, 
		stride, 
		num_filter) in zip(dilation_rates, 
							range(num_layers), 
							kernel_sizes,
							strides, 
							num_filters):
		layer_scope_name = "%s_layer_%s"%(str(scope_name), str(layer))
		output_shape = bert_utils.get_shape_list(inputs, expected_rank=3)
		with tf.variable_scope(layer_scope_name, reuse=reuse):
			if dilation_rate > 1:
				stride = 1
			if not is_casual:
				padding = padding
				tf.logging.info("==none-casual same padding==")
			else:
				left_pad = dilation_rate * (kernel_size - 1)
				inputs = tf.pad(inputs, [[0, 0, ], [left_pad, 0], [0, 0]])
				padding = 'VALID'
				tf.logging.info("==casual valid padding==")

			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			gatedcnn_outputs = residual_gated_conv1d_op(inputs,
									residual_inputs,
									filters=num_filter, 
									kernel_size=kernel_size, 
									padding=padding, 
									activation=None, 
									strides=stride, 
									reuse=False, 
									dilation_rate=dilation_rate,
									name="residual_gated_conv",
									kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1), 
									is_training=is_training)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("intermediate"):
				intermediate_output = tf.layers.dense(
						gatedcnn_outputs,
						num_filter*4,
						activation=tf.nn.relu,
						kernel_initializer=create_initializer(0.02))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope("output"):
				layer_output = tf.layers.dense(
						intermediate_output,
						num_filter,
						kernel_initializer=create_initializer(0.02))

			layer_output = tf.nn.dropout(layer_output, 1-dropout_rate)
			inputs = layer_norm(layer_output + gatedcnn_outputs)

			if padding == 'SAME':
				inputs *= input_mask
			residual_inputs = inputs
	
	return inputs

def backward_dgcnn(x, input_mask,
			num_layers=2, 
			dilation_rates=[1,2],
			strides=[1,1],
			num_filters=[64,64],
			kernel_sizes=[3,3], 
			is_training=False,
			scope_name="textcnn", 
			reuse=False, 
			activation=tf.nn.relu,
			is_casual=False,
			padding='SAME'):

	# input_mask: batch_size, seq

	# initializer = tf.glorot_uniform_initializer()
	# initializer = tf.truncated_normal_initializer(stddev=0.1)
	initializer = create_initializer(initializer_range=0.02)
	input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), axis=-1)

	# inverse_mask = tf.reverse_sequence(input_mask, input_len, seq_axis=1, batch_axis=0)
	input_mask = tf.expand_dims(input_mask, axis=-1)
	input_mask = tf.cast(input_mask, dtype=tf.float32)

	inverse_x = tf.reverse_sequence(x, input_len, seq_axis=1, batch_axis=0)

	if is_casual:
		left_pad = dilation_rates[0] * (kernel_sizes[0] - 1)
		inputs = tf.pad(inverse_x, [[0, 0, ], [left_pad, 0], [0, 0]])
		padding = 'VALID'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = inverse_x
		padding = 'SAME'

	with tf.variable_scope(scope_name, reuse=reuse):
		inputs = gated_conv1d_op(inputs,
						filters=num_filters[0],
						kernel_size=kernel_sizes[0],
						padding=padding,
						activation=None,
						strides=1,
						reuse=reuse, 
						dilation_rate=1,
						name="gated_conv",
						kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1),
						is_training=is_training)
		if padding == 'SAME':
			inputs *= input_mask
		residual_inputs = inputs

	for (dilation_rate, 
		layer, 
		kernel_size, 
		stride, 
		num_filter) in zip(dilation_rates, 
							range(num_layers), 
							kernel_sizes,
							strides, 
							num_filters):
		layer_scope_name = "%s_layer_%s"%(str(scope_name), str(layer))
		output_shape = bert_utils.get_shape_list(inputs, expected_rank=3)
		with tf.variable_scope(layer_scope_name, reuse=reuse):
			if dilation_rate > 1:
				stride = 1
			if not is_casual:
				padding = padding
				tf.logging.info("==none-casual same padding==")
			else:
				left_pad = dilation_rate * (kernel_size - 1)
				inputs = tf.pad(inputs, [[0, 0, ], [left_pad, 0], [0, 0]])
				padding = 'VALID'
				tf.logging.info("==casual valid padding==")

			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			gatedcnn_outputs = residual_gated_conv1d_op(inputs,
									residual_inputs,
									filters=num_filter, 
									kernel_size=kernel_size, 
									padding=padding, 
									activation=None, 
									strides=stride, 
									reuse=False, 
									dilation_rate=dilation_rate,
									name="residual_gated_conv",
									kernel_initializer=initializer, #tf.truncated_normal_initializer(stddev=0.1), 
									is_training=is_training)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("intermediate"):
				intermediate_output = tf.layers.dense(
						gatedcnn_outputs,
						num_filter*4,
						activation=tf.nn.relu,
						kernel_initializer=create_initializer(0.02))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope("output"):
				layer_output = tf.layers.dense(
						intermediate_output,
						num_filter,
						kernel_initializer=create_initializer(0.02))

			layer_output = tf.nn.dropout(layer_output, 1-dropout_rate)
			inputs = layer_norm(layer_output + gatedcnn_outputs)

			if padding == 'SAME':
				inputs *= input_mask
			residual_inputs = inputs

	inverse_x = tf.reverse_sequence(inputs, input_len, seq_axis=1, batch_axis=0)
	return inputs

