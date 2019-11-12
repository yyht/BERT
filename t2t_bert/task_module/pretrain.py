from utils.bert import bert_utils
from utils.bert import bert_modules

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

	scope = kargs.get('scope', None)
	if scope:
		scope = scope + '/' + 'cls/predictions'
	else:
		scope = 'cls/predictions'

	tf.logging.info("**** mlm scope **** %s", str(scope))

	input_tensor = bert_utils.gather_indexes(input_tensor, positions)
	"""
	flatten masked lm ids with positions
	"""
	# with tf.variable_scope("cls/predictions", reuse=reuse):
	with tf.variable_scope(scope, reuse=reuse):
		# We apply one more non-linear transformation before the output layer.
		# This matrix is not used after pre-training.
		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=config.hidden_size,
					activation=bert_modules.get_activation(config.hidden_act),
					kernel_initializer=bert_modules.create_initializer(
							config.initializer_range))
			input_tensor = bert_modules.layer_norm(input_tensor)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		output_bias = tf.get_variable(
				"output_bias",
				shape=[config.vocab_size],
				initializer=tf.zeros_initializer())
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

def get_next_sentence_output(config, input_tensor, labels, reuse=None, **kargs):
	"""Get loss and log probs for the next sentence prediction."""
	# Simple binary classification. Note that 0 is "next sentence" and 1 is
	# "random sentence". This weight matrix is not used after pre-training.

	scope = kargs.get('scope', None)
	if scope:
		scope = scope + '/' + 'cls/seq_relationship'
	else:
		scope = 'cls/seq_relationship'
	tf.logging.info("**** nsp scope **** %s", str(scope))

	# with tf.variable_scope("cls/seq_relationship", reuse=reuse):
	with tf.variable_scope(scope, reuse=reuse):
		output_weights = tf.get_variable(
				"output_weights",
				shape=[2, config.hidden_size],
				initializer=bert_modules.create_initializer(config.initializer_range))
		output_bias = tf.get_variable(
				"output_bias", shape=[2], initializer=tf.zeros_initializer())

		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		labels = tf.reshape(labels, [-1])
		one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, per_example_loss, log_probs)

def seq_mask_masked_lm_output(config, input_tensor, output_weights,
							input_mask, input_ori_ids, input_ids, 
							sampled_binary_mask, **kargs):

	input_shape_list = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]
	hidden_dims = input_shape_list[2]

	embedding_projection = kargs.get('embedding_projection', None)

	scope = kargs.get('scope', None)
	if scope:
		scope = scope + '/' + 'cls/predictions'
	else:
		scope = 'cls/predictions'

	tf.logging.info("**** mlm generator scope **** %s", str(scope))

	# with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		if config.get('ln_type', 'postln') == 'preln':
			input_tensor = bert_modules.layer_norm(input_tensor)
		elif config.get('ln_type', 'postln') == 'postln':
			input_tensor = input_tensor
		else:
			input_tensor = input_tensor

		if config.get("embedding", "factorized") == "factorized":
			projection_width = config.hidden_size
		else:
			projection_width = config.embedding_size

		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=projection_width,
					activation=bert_modules.get_activation(config.hidden_act),
					kernel_initializer=bert_modules.create_initializer(
							config.initializer_range))

			if config.get('ln_type', 'postln') == 'preln':
				input_tensor = input_tensor
			elif config.get('ln_type', 'postln') == 'postln':
				input_tensor = bert_modules.layer_norm(input_tensor)
			else:
				input_tensor = bert_modules.layer_norm(input_tensor)

		if embedding_projection is not None:
			# batch x seq x hidden, embedding x hidden
			print(input_tensor.get_shape(), embedding_projection.get_shape())
			input_tensor = tf.einsum("abc,dc->abd", input_tensor, embedding_projection)
		else:
			print("==no need for embedding projection==")
			input_tensor = input_tensor

		output_bias = tf.get_variable(
				"output_bias",
				shape=[config.vocab_size],
				initializer=tf.zeros_initializer())
		# batch x seq x embedding
		logits = tf.einsum("abc,dc->abd", input_tensor, output_weights)
		logits = tf.nn.bias_add(logits, output_bias)

		"""
		if input_ori_ids[i] is random pertubated, sampled_binary_mask[i]=1
		"""
		sampled_binary_mask = tf.cast(sampled_binary_mask, tf.float32)
		input_mask = tf.cast(input_mask, tf.float32)

		sampled_binary_mask *= input_mask

		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits,
												labels=tf.stop_gradient(input_ori_ids),
												)
		per_example_loss *= sampled_binary_mask
		loss = tf.reduce_sum(per_example_loss) / tf.reduce_sum(sampled_binary_mask)

		return (loss, per_example_loss, logits, sampled_binary_mask)





