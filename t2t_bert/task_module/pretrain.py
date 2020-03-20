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

	# if config.get("embedding", "factorized") == "factorized":
	# 	projection_width = config.hidden_size
	# else:
	# 	projection_width = config.embedding_size

	if config.get("embedding", "none_factorized") == "none_factorized":
		projection_width = config.hidden_size
		tf.logging.info("==not using embedding factorized==")
	else:
		projection_width = config.get('embedding_size', config.hidden_size)
		tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

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
					units=projection_width,
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

		# if config.get("embedding", "factorized") == "factorized":
		# 	projection_width = config.hidden_size
		# else:
		# 	projection_width = config.embedding_size

		if config.get("embedding", "none_factorized") == "none_factorized":
			projection_width = config.hidden_size
			tf.logging.info("==not using embedding factorized==")
		else:
			projection_width = config.get('embedding_size', config.hidden_size)
			tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

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
		loss = tf.reduce_sum(per_example_loss) / (1e-10 + tf.reduce_sum(sampled_binary_mask))

		return (loss, per_example_loss, logits, sampled_binary_mask)

def emb_score(config, input_tensor, input_ids, 
				output_weights,
				input_mask, **kargs):

	input_shape_list = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]
	hidden_dims = input_shape_list[2]

	scope = kargs.get('scope', None)
	if scope:
		lm_scope = scope + '/' + 'cls/predictions'
	else:
		lm_scope = 'cls/predictions'

	tf.logging.info("**** mlm generator scope **** %s", str(lm_scope))

	# with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
	with tf.variable_scope(lm_scope, reuse=tf.AUTO_REUSE):
		if config.get('ln_type', 'postln') == 'preln':
			input_tensor = bert_modules.layer_norm(input_tensor)
		elif config.get('ln_type', 'postln') == 'postln':
			input_tensor = input_tensor
		else:
			input_tensor = input_tensor

		if config.get("embedding", "none_factorized") == "none_factorized":
			projection_width = config.hidden_size
			tf.logging.info("==not using embedding factorized==")
		else:
			projection_width = config.get('embedding_size', config.hidden_size)
			tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

		if kargs.get("energy_pooling", "mi") == "mi":
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
			tf.logging.info("****** mi using mlm transform *******")
		else:
			with tf.variable_scope("transform_ebm"):
				input_tensor = tf.layers.dense(
				  input_tensor,
				  units=projection_width,
				  activation=tf.nn.tanh,
				  kernel_initializer=bert_modules.create_initializer(
					  config.initializer_range))
			tf.logging.info("****** using other pooling transform *******")

	# with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
	if scope:
		ebm_scope = scope + '/' + 'ebm/predictions'
	else:
		ebm_scope = 'ebm/predictions'
	
	tf.logging.info("**** ebm generator scope **** %s", str(ebm_scope))

	print(input_tensor.get_shape(), "==input_tensor shape==")

	with tf.variable_scope(ebm_scope, reuse=tf.AUTO_REUSE):
		# assume the whole model is self-normalization

		if kargs.get("normalized_constant", "constant") == 'constant':
			normalized_constant = tf.get_variable(
					"ebm_normalized_constant",
					shape=[config.max_position_embeddings],
					initializer=tf.zeros_initializer())
			tf.logging.info("****** cosntant logz *******")
		elif kargs.get("normalized_constant", "length_linear") == 'length_linear':
			normalized_constant = tf.get_variable(
					"ebm_normalized_constant",
					shape=[config.max_position_embeddings],
					initializer=tf.constant_initializer(np.arange((config.max_position_embeddings))+1, tf.float32),
					trainable=False)
			scale_weights = tf.get_variable(
					"ebm_normalized_constant_scale",
					shape=[config.max_position_embeddings],
					initializer=tf.constant_initializer(np.log(config.vocab_size)*np.ones((config.max_position_embeddings)), dtype=tf.float32),
					trainable=True)
			scale_bias = tf.get_variable(
					"ebm_normalized_constant_bias",
					shape=[config.max_position_embeddings],
					initializer=tf.zeros_initializer(),
					trainable=True)
			tf.logging.info("****** length linear logz *******")
			# normalized_constant = scale_bias + scale_weights * tf.pow(normalized_constant, 2)

		valid_seq_length = tf.cast(tf.reduce_sum(input_mask, axis=-1), tf.int32) # batch_size
		onehot_length_ids = tf.one_hot(valid_seq_length, config.max_position_embeddings)
		
		length_part = tf.einsum("ab,b->a", tf.cast(onehot_length_ids, tf.float32), normalized_constant)
		length_scale_part = tf.einsum("ab,b->a", tf.cast(onehot_length_ids, tf.float32), scale_weights)
		length_bias_part = tf.einsum("ab,b->a", tf.cast(onehot_length_ids, tf.float32), scale_bias)

		input_normalized_constant = length_part*length_scale_part + length_bias_part

		# input_normalized_constant = tf.einsum("ab,b->a", tf.cast(onehot_length_ids, tf.float32), normalized_constant)

		# f_input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1), tf.float32)

		if kargs.get("energy_pooling", "mi") == "mean_pooling":
			tf.logging.info("==apply mean pooling to get hidden states projections==")
			# for input token sequence: <start> a b c
			# we only calculate energy on a,b,c which <start> can't contribute to final 
			# energy function
			# batch x dim
			pool_features = tf.einsum("abc,ab->ac", input_tensor[:, 1:], tf.cast(input_mask[:, 1:], tf.float32))
			pool_features /= (1e-10+tf.reduce_sum(tf.cast(input_mask[:, 1:], tf.float32), axis=1, keepdims=True))
			# tf.reduce_sum(input_tensor*f_input_mask, axis=1) #/ (1e-10+tf.reduce_sum(f_input_mask, axis=1))

			print(pool_features.get_shape(), "===pool_features shape===")
		elif kargs.get("energy_pooling", "mi") == "mi":
			tf.logging.info("==apply mi to get hidden states projections==")
			# input_tensor_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(input_tensor, 2), axis=-1))+1e-20, axis=-1)
			# input_tensor = input_tensor / tf.stop_gradient(input_tensor_norm)
			# output_weights_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(output_weights, 2), axis=-1))+1e-20, axis=-1)
			# output_weights = output_weights / tf.stop_gradient(output_weights_norm)
			# we calculate cosine distance to make mi bounded by [-1, 1]
			logits = tf.einsum("abc,dc->abd", input_tensor, output_weights) # batch x seq x vocab

			input_id_shape = bert_utils.get_shape_list(input_ids, [2,3])
			if len(input_id_shape) == 2:
				onehot_input_ids = tf.cast(tf.one_hot(tf.cast(input_ids, tf.int32), config.vocab_size), tf.float32) # batch x seq x vocab
				input_ori_ids = tf.cast(onehot_input_ids, tf.float32)
				print("==input ori ids shape== 2-dim", input_ori_ids.get_shape())
			else:
				input_ori_ids = tf.cast(input_ids, tf.float32)
				print("==input ori ids shape== 3-dim", input_ori_ids.get_shape())

			logits = tf.einsum("abd,abd->ab", logits, input_ori_ids)
			print(logits.get_shape(), "==pooled logits shape==")
			# with l2-normalize, we can bound logits to 1
			pool_features = tf.reduce_sum(logits[:, 1:]*tf.cast(input_mask[:, 1:], tf.float32), axis=1) #/ (1e-10+tf.reduce_sum(tf.cast(input_mask[:, 1:], tf.float32), axis=1))
			pool_features = tf.expand_dims(pool_features, axis=-1)
			print(pool_features.get_shape(), "==pooled feature shape==")

			if kargs.get("softplus_features", False):
				# when pooled_features is to infinite, it converges to 0
				# when is to minus inifinite, it will converges to inifite
				pool_features = tf.nn.softplus(-pool_features)
				tf.logging.info("****** apply softplus transformation for pooled_features *******")

		# batch_size x hidden_dims

		if kargs.get('transform', True):

			if kargs.get("transformer_activation", "none") == 'softplus':
				with tf.variable_scope("transform"):
					ebm_scalar = tf.layers.dense(
							pool_features,
							units=1,
							use_bias=True,
							activation=tf.nn.softplus # mask scalar to [0,inifite]
							)
				tf.logging.info("****** apply softplus *******")
			elif kargs.get("transformer_activation", "none") == 'linear':
				tf.logging.info("****** apply linear projection *******")
				with tf.variable_scope("transform"):
					ebm_scalar = tf.layers.dense(
							pool_features,
							units=1,
							use_bias=False,
							activation=None # mask scalar to [0,inifite]
							)
			else:
				with tf.variable_scope("transform"):

					feature_shape = bert_utils.get_shape_list(pool_features, expected_rank=[1,2])

					pool_features = tf.layers.dense(
							pool_features,
							units=feature_shape[-1],
							activation=tf.nn.relu,
							)

					output_weights = tf.get_variable(
							"output_weights", [config.max_position_embeddings, feature_shape[-1]],
							initializer=tf.truncated_normal_initializer(stddev=0.02))

					output_bias = tf.get_variable(
							"output_bias", [config.max_position_embeddings], 
							initializer=tf.constant_initializer(-np.log(np.arange(config.max_position_embeddings).astype(np.float32)+1.0), dtype=tf.float32)
							)
				
					# batch x max_position_embeddings
					ebm_scalar_pos = tf.nn.relu(tf.matmul(pool_features, output_weights, transpose_b=True)) + output_bias
					
					pos_tensor = tf.cast(tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1), tf.int32)
					onehot_pos = tf.cast(tf.one_hot(tf.cast(pos_tensor, tf.int32), config.max_position_embeddings), tf.float32) # batch x seq x vocab
					ebm_scalar = tf.einsum("ab,ab->a", ebm_scalar_pos, onehot_pos)
					ebm_scalar = tf.expand_dims(ebm_scalar, axis=-1)

				tf.logging.info("****** apply linear projection *******")
			print("===ebm_scalar====", ebm_scalar.get_shape())

			ebm_scalar = tf.squeeze(ebm_scalar, axis=-1)
			print("===ebm_scalar====", ebm_scalar.get_shape())
			# ebm_scalar /= (1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
			
			# if kargs.get("energy_pooling", "mi") == "mean_pooling":
			
			print("===ebm_scalar====", ebm_scalar.get_shape())
			print("===input_normalized_constant====", input_normalized_constant.get_shape())

		else:
			ebm_scalar = tf.squeeze(pool_features, axis=-1)
			# ebm_scalar /= (1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
			print("===ebm_scalar====", ebm_scalar.get_shape())
			print("===input_normalized_constant====", input_normalized_constant.get_shape())

		if not kargs.get("prob_ln", False):
			tf.logging.info("****** sum of plogprob as sentence probability *******")
			# ebm_scalar /= (1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
		else:
			ebm_scalar /= (1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
			tf.logging.info("****** sum of plogprob with length normalization as sentence probability *******")
		print("===ebm_scalar====", ebm_scalar.get_shape())
		print("===input_normalized_constant====", input_normalized_constant.get_shape())

		# original ebm log-likelihood:
		# log(exp(-E(x))/Z) = -E(x) - log(Z)
		# here we use bert encoder of pooled hidden states as energy function which need to minus when apply to 
		# actual energy function

		if kargs.get("logz_mode", "default") == 'default':
			tf.logging.info("****** default logz *******")
			logits = -ebm_scalar - input_normalized_constant - tf.log(1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
		elif kargs.get("logz_mode", "default") == 'standard':
			logits = ebm_scalar - input_normalized_constant
			tf.logging.info("****** standard logz *******")
		elif kargs.get("logz_mode", "default") == 'standard_minus':
			tf.logging.info("****** minus standard logz *******")
			logits = -ebm_scalar - input_normalized_constant
		elif kargs.get("logz_mode", "default") == 'constant':
			logits = -ebm_scalar - tf.log(1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))
			tf.logging.info("****** constant logz *******")
		elif kargs.get("logz_mode", "self_normalizing") == 'self_normalizing':
			logits = -ebm_scalar
			tf.logging.info("****** self_normalizing *******")
		else:
			tf.logging.info("****** linear logz *******")
			logits = ebm_scalar - input_normalized_constant * tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1)
		print("=ebm logits shape==", logits.get_shape())

	return logits

