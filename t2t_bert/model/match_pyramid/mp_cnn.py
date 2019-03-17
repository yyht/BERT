import tensorflow as tf
import numpy as np

def get_activation(activation_type):
	if activation_type == "relu":
		return tf.nn.relu
	elif activation_type == "sigmoid":
		return tf.nn.sigmoid
	elif activation_type == "tanh":
		return tf.nn.tanh

def _mp_cnn_layer(config, cross, dpool_index, filters, kernel_size, pool_size, strides, name, reuse=None):
	cross_conv = tf.layers.conv2d(
		inputs=cross,
		filters=filters,
		kernel_size=kernel_size,
		padding="same",
		activation=get_activation(config["mp_activation"]),
		strides=1,
		reuse=reuse,
		name=name+"cross_conv")
	if config["mp_dynamic_pooling"] and dpool_index is not None:
		cross_conv = tf.gather_nd(cross_conv, dpool_index)
	cross_pool = tf.layers.max_pooling2d(
		inputs=cross_conv,
		pool_size=pool_size,
		strides=strides,
		padding="valid",
		name=name+"cross_pool")
	return cross_pool


def _mp_semantic_feature_layer(config, match_matrix, dpool_index, reuse=None):

	# conv-pool layer 1
	filters = config["mp_num_filters"][0]
	kernel_size = config["mp_filter_sizes"][0]
	seq_len = config["max_seq_len"]
	pool_size0 = config["mp_pool_sizes"][0]
	pool_sizes = [seq_len / pool_size0, seq_len / pool_size0]
	strides = [seq_len / pool_size0, seq_len / pool_size0]
	conv1 = _mp_cnn_layer(match_matrix, dpool_index, filters, kernel_size, pool_sizes, strides, 
							name=model_name+"1", reuse=reuse)
	conv1_flatten = tf.reshape(conv1, [-1, config["mp_num_filters"][0] * (pool_size0 * pool_size0)])

	# conv-pool layer 2
	filters = config["mp_num_filters"][1]
	kernel_size = config["mp_filter_sizes"][1]
	pool_size1 = config["mp_pool_sizes"][1]
	pool_sizes = [pool_size0 / pool_size1, pool_size0 / pool_size1]
	strides = [pool_size0 / pool_size1, pool_size0 / pool_size1]
	conv2 = _mp_cnn_layer(conv1, None, filters, kernel_size, pool_sizes, strides, name=model_name + "2",
							reuse=reuse)
	conv2_flatten = tf.reshape(conv2, [-1, config["mp_num_filters"][1] * (pool_size1 * pool_size1)])

	# cross = tf.concat([conv1_flatten, conv2_flatten], axis=-1)

	return conv2_flatten