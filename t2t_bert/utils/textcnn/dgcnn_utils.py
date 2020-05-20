import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers
from utils.bert import bert_utils
import tensorflow.contrib.layers as layers

"""
https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/rnn.py#L360-L514
https://github.com/allenai/bilm-tf/blob/master/bilm/model.py
https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/gated_cnn_encoder.py
"""

def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)

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
	conv_linear = tf.layers.conv1d(
		inputs=inputs,
		filters=filters,
		kernel_size=kernel_size,
		padding=padding,
		activation=None,
		strides=strides,
		reuse=reuse,
		name=name+"_linear",
		kernel_initializer=kernel_initializer,
		dilation_rate=dilation_rate)
	conv_gated = tf.layers.conv1d(
		inputs=inputs,
		filters=filters,
		kernel_size=kernel_size,
		padding=padding,
		activation=tf.nn.sigmoid,
		strides=strides,
		reuse=reuse,
		name=name+"_gated",
		kernel_initializer=kernel_initializer,
		dilation_rate=dilation_rate)
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
	conv_linear = tf.layers.conv1d(
		inputs=inputs,
		filters=filters,
		kernel_size=kernel_size,
		padding=padding,
		activation=None,
		strides=strides,
		reuse=reuse,
		name=name+"_linear",
		kernel_initializer=kernel_initializer,
		dilation_rate=dilation_rate)
	conv_gated = tf.layers.conv1d(
		inputs=inputs,
		filters=filters,
		kernel_size=kernel_size,
		padding=padding,
		activation=None,
		strides=strides,
		reuse=reuse,
		name=name+"_gated",
		kernel_initializer=kernel_initializer,
		dilation_rate=dilation_rate)
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
			padding='same'
			):

	# input_mask: batch_size, seq

	# initializer = tf.glorot_uniform_initializer()
	initializer = create_initializer(initializer_range=0.02)

	input_mask = tf.cast(input_mask, dtype=tf.float32)
	input_mask = tf.expand_dims(input_mask, axis=-1)

	if is_casual:
		left_pad = dilation_rates[0] * (kernel_sizes[0] - 1)
		inputs = tf.pad(x, [[0, 0, ], [left_pad, 0], [0, 0]])
		padding = 'valid'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = x

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
		if padding == 'same':
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
				padding = 'valid'
				tf.logging.info("==casual valid padding==")

			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			inputs = residual_gated_conv1d_op(inputs,
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
			if padding == 'same':
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
			padding='same'):

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
		padding = 'valid'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = inverse_x

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
		if padding == 'same':
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
				padding = 'valid'
				tf.logging.info("==casual valid padding==")

			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			inputs = residual_gated_conv1d_op(inputs,
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
			if padding == 'same':
				inputs *= input_mask
			residual_inputs = inputs
	inverse_x = tf.reverse_sequence(inputs, input_len, seq_axis=1, batch_axis=0)
	return inputs