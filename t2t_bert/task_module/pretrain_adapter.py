from utils.bert import bert_utils
from utils.bert import bert_adapter_modules

import numpy as np

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

def get_masked_lm_output(config, input_tensor, output_weights, positions,
							label_ids, label_weights, **kargs):
	"""Get loss and log probs for the masked LM."""
	reuse = kargs.get('reuse', False)
	input_tensor = tf.cast(input_tensor, tf.float32)
	positions = tf.cast(positions, tf.int32)
	label_ids = tf.cast(label_ids, tf.int32)
	label_weights = tf.cast(label_weights, tf.float32)

	input_tensor = bert_utils.gather_indexes(input_tensor, positions)
	"""
	flatten masked lm ids with positions
	"""
	with tf.variable_scope("cls/predictions", reuse=reuse):
		# We apply one more non-linear transformation before the output layer.
		# This matrix is not used after pre-training.
		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=config.hidden_size,
					activation=bert_adapter_modules.get_activation(config.hidden_act),
					kernel_initializer=bert_adapter_modules.create_initializer(
							config.initializer_range),
					variables_collections=["heads", tf.GraphKeys.GLOBAL_VARIABLES])
			input_tensor = bert_adapter_modules.layer_norm(input_tensor)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		output_bias = tf.get_variable(
				"output_bias",
				shape=[config.vocab_size],
				initializer=tf.zeros_initializer(),
				variables_collections=["heads", tf.GraphKeys.GLOBAL_VARIABLES])
		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)

		label_ids = tf.reshape(label_ids, [-1])
		label_weights = tf.reshape(label_weights, [-1])

		one_hot_labels = tf.one_hot(
				label_ids, depth=config.vocab_size, dtype=tf.float32)

		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													labels=tf.stop_gradient(label_ids),
													logits=logits)
		# per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

		numerator = tf.reduce_sum(label_weights * per_example_loss)
		denominator = tf.reduce_sum(label_weights) + 1e-5

		# The `positions` tensor might be zero-padded (if the sequence is too
		# short to have the maximum number of predictions). The `label_weights`
		# tensor has a value of 1.0 for every real prediction and 0.0 for the
		# padding predictions.
		# per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
		# numerator = tf.reduce_sum(label_weights * per_example_loss)
		# denominator = tf.reduce_sum(label_weights) + 1e-5
		loss = numerator / denominator

	return (loss, per_example_loss, log_probs, label_weights)

def get_next_sentence_output(config, input_tensor, labels, reuse=None):
	"""Get loss and log probs for the next sentence prediction."""
	# Simple binary classification. Note that 0 is "next sentence" and 1 is
	# "random sentence". This weight matrix is not used after pre-training.
	with tf.variable_scope("cls/seq_relationship", reuse=reuse):
		output_weights = tf.get_variable(
				"output_weights",
				shape=[2, config.hidden_size],
				initializer=bert_adapter_modules.create_initializer(config.initializer_range),
				variables_collections=["heads", tf.GraphKeys.GLOBAL_VARIABLES])
		output_bias = tf.get_variable(
				"output_bias", shape=[2], initializer=tf.zeros_initializer(),
				variables_collections=["heads", tf.GraphKeys.GLOBAL_VARIABLES])

		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		labels = tf.reshape(labels, [-1])
		one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, per_example_loss, log_probs)


