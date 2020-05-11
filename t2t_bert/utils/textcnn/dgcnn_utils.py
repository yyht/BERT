import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers
from utils.bert import bert_utils
import tensorflow.contrib.layers as layers

def residual_gated_conv1d_op(inputs, 
							residual_inputs,
							filters=8, kernel_size=3, 
							padding="same", 
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

# def conv_shape_compute(input_shape, num_filter,  dilation_rate, 
# 			kernel_size, strides):
# 	input_shape[1] = int((input_shape[1]-dilation_rate*(kernel_size))/strides+dilation_rate)
# 	input_shape[-1] = num_filter
# 	return input_shape

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

	input_mask = tf.cast(input_mask, dtype=tf.float32)
	input_mask = tf.expand_dims(input_mask, axis=-1)

	residual_inputs = x
	inputs = x
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
									kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), 
									is_training=is_training)
			if padding == 'same':
				inputs *= input_mask
			residual_inputs = inputs
	return inputs

