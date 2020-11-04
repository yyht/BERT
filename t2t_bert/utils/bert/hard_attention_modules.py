from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np

from utils.bert import bert_utils
from utils.bert import layer_norm_utils
from utils.bert import dropout_utils
from utils.bert import bert_modules

def gelu(input_tensor):
	"""Gaussian Error Linear Unit.

	This is a smoother version of the RELU.
	Original paper: https://arxiv.org/abs/1606.08415

	Args:
		input_tensor: float Tensor to perform activation.

	Returns:
		`input_tensor` with the GELU activation applied.
	"""
	cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))

def attention_layer(from_tensor,
										to_tensor,
										attention_mask=None,
										num_attention_heads=1,
										size_per_head=512,
										query_act=None,
										key_act=None,
										value_act=None,
										attention_probs_dropout_prob=0.0,
										initializer_range=0.02,
										do_return_2d_tensor=False,
										batch_size=None,
										from_seq_length=None,
										to_seq_length=None,
										attention_fixed_size=None,
										dropout_name=None,
										structural_attentions="none",
										is_training=False,
										**kargs):
	"""Performs multi-headed attention from `from_tensor` to `to_tensor`.

	This is an implementation of multi-headed attention based on "Attention
	is all you Need". If `from_tensor` and `to_tensor` are the same, then
	this is self-attention. Each timestep in `from_tensor` attends to the
	corresponding sequence in `to_tensor`, and returns a fixed-with vector.

	This function first projects `from_tensor` into a "query" tensor and
	`to_tensor` into "key" and "value" tensors. These are (effectively) a list
	of tensors of length `num_attention_heads`, where each tensor is of shape
	[batch_size, seq_length, size_per_head].

	Then, the query and key tensors are dot-producted and scaled. These are
	softmaxed to obtain attention probabilities. The value tensors are then
	interpolated by these probabilities, then concatenated back to a single
	tensor and returned.

	In practice, the multi-headed attention are done with transposes and
	reshapes rather than actual separate tensors.

	Args:
		from_tensor: float Tensor of shape [batch_size, from_seq_length,
			from_width].
		to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
		attention_mask: (optional) int32 Tensor of shape [batch_size,
			from_seq_length, to_seq_length]. The values should be 1 or 0. The
			attention scores will effectively be set to -infinity for any positions in
			the mask that are 0, and will be unchanged for positions that are 1.
		num_attention_heads: int. Number of attention heads.
		size_per_head: int. Size of each attention head.
		query_act: (optional) Activation function for the query transform.
		key_act: (optional) Activation function for the key transform.
		value_act: (optional) Activation function for the value transform.
		attention_probs_dropout_prob:
		initializer_range: float. Range of the weight initializer.
		do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
			* from_seq_length, num_attention_heads * size_per_head]. If False, the
			output will be of shape [batch_size, from_seq_length, num_attention_heads
			* size_per_head].
		batch_size: (Optional) int. If the input is 2D, this might be the batch size
			of the 3D version of the `from_tensor` and `to_tensor`.
		from_seq_length: (Optional) If the input is 2D, this might be the seq length
			of the 3D version of the `from_tensor`.
		to_seq_length: (Optional) If the input is 2D, this might be the seq length
			of the 3D version of the `to_tensor`.

	Returns:
		float Tensor of shape [batch_size, from_seq_length,
			num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
			true, this will be of shape [batch_size * from_seq_length,
			num_attention_heads * size_per_head]).

	Raises:
		ValueError: Any of the arguments or tensor shapes are invalid.
	"""

	def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
													 seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])

		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor

	from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
	to_shape = bert_utils.get_shape_list(to_tensor, expected_rank=[2, 3])

	if len(from_shape) != len(to_shape):
		raise ValueError(
				"The rank of `from_tensor` must match the rank of `to_tensor`.")

	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
		to_seq_length = to_shape[1]
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None or to_seq_length is None):
			raise ValueError(
					"When passing in rank 2 tensors to attention_layer, the values "
					"for `batch_size`, `from_seq_length`, and `to_seq_length` "
					"must all be specified.")

	# Scalar dimensions referenced here:
	#   B = batch size (number of sequences)
	#   F = `from_tensor` sequence length
	#   T = `to_tensor` sequence length
	#   N = `num_attention_heads`
	#   H = `size_per_head`

	if attention_fixed_size:
		attention_head_size = attention_fixed_size
		tf.logging.info("==apply attention_fixed_size==")
	else:
		attention_head_size = size_per_head
		tf.logging.info("==apply attention_original_size==")

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
	to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

	# `query_layer` = [B*F, N*H]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=query_act,
			name="query",
			kernel_initializer=bert_modules.create_initializer(initializer_range))

	# `key_layer` = [B*T, N*H]
	key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=key_act,
			name="key",
			kernel_initializer=bert_modules.create_initializer(initializer_range))

	# `value_layer` = [B*T, N*H]
	value_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=value_act,
			name="value",
			kernel_initializer=bert_modules.create_initializer(initializer_range))

	# `query_layer` = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size,
									 num_attention_heads, 
									 from_seq_length,
									 attention_head_size)

	# `key_layer` = [B, N, T, H]
	key_layer = transpose_for_scores(key_layer, batch_size, 
									num_attention_heads,
									to_seq_length, 
									attention_head_size)

	# Take the dot product between "query" and "key" to get the raw
	# attention scores.
	# `attention_scores` = [B, N, F, T]
	attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
	attention_scores = tf.multiply(attention_scores,
									1.0 / math.sqrt(float(attention_head_size)))

	# `value_layer` = [B, T, N, H]
	value_layer = tf.reshape(
			value_layer,
			[batch_size, to_seq_length, num_attention_heads, attention_head_size])

	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		value_layer = tf.reshape(
				value_layer,
				[batch_size * to_seq_length, num_attention_heads * attention_head_size])
	else:
		# `context_layer` = [B, F, N*V]
		value_layer = tf.reshape(
				value_layer,
				[batch_size, to_seq_length, num_attention_heads * attention_head_size])

	return attention_scores, value_layer

def hard_attention(attention_scores,
					value_output,
					attention_mask, 
					batch_size,
					from_seq_length,
					to_seq_length, 
					num_attention_heads,
					attention_head_size,
					do_return_2d_tensor=False):
	# [batch_size, to_seq_length, num_attention_heads, attention_head_size]
	value_output = tf.reshape(
	value_output, [batch_size, to_seq_length, 
					num_attention_heads, 
					attention_head_size])
	# [batch_size, num_attention_heads, to_seq_length, attention_head_size]
	value_output = tf.transpose(value_output, [0,2,1,3])
	# [batch_size, num_attention_heads, to_seq_length]
	value_output_norm = tf.norm(
		value_output, ord='euclidean', axis=-1
	)

	# [batch_size, 1, to_seq_length]
	value_mask = attention_mask[:, 0:1, :]
	# [batch_size, 1, 1]
	value_len = 1e-10+tf.cast(tf.reduce_sum(value_mask, axis=-1, keep_dims=True), dtype=tf.float32)
	# [batch_size, 1, to_seq_length]
	value_adder = (1.0 - tf.cast(value_mask, tf.float32)) * -10000.0

	# Since we are adding it to the raw scores before the softmax, this is
	# effectively the same as removing these entirely.
	# [batch_size, num_attention_heads, seq_length]
	value_output_norm_prob = tf.nn.softmax(value_output_norm+value_adder)
	threshold = tf.cast(value_len * num_attention_heads, dtype=tf.float32)
	# [batch_size, num_attention_heads, to_seq_length]
	norm_mask = tf.cast(tf.greater(value_output_norm_prob, 1.0 / threshold), dtype=tf.float32)
	# [batch_size, num_attention_heads, 1, to_seq_length]
	value_output_mask = tf.expand_dims(norm_mask, axis=[2])
	# [batch_size, 1, from_seq_length, to_seq_length]
	multihead_attention_mask = tf.expand_dims(attention_mask, axis=[1])
	multihead_attention_mask = tf.cast(multihead_attention_mask, dtype=tf.float32)
	multihead_attention_mask *= value_output_mask

	attention_adder = (1.0 - multihead_attention_mask) * -10000.0

	# Since we are adding it to the raw scores before the softmax, this is
	# effectively the same as removing these entirely.
	# [B, N, F, T]
	attention_scores += attention_adder
	attention_probs = tf.exp(tf.nn.log_softmax(attention_scores))
	
	# `context_layer` = [B, N, F, H]
	context_layer = tf.matmul(attention_probs, value_output)

	# `context_layer` = [B, F, N, H]
	context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		context_layer = tf.reshape(
				value_layer,
				[batch_size * from_seq_length, num_attention_heads * attention_head_size])
	else:
		# `context_layer` = [B, F, N*V]
		context_layer = tf.reshape(
				value_layer,
				[batch_size, from_seq_length, num_attention_heads * attention_head_size])

	return attention_scores, value_layer

def transformer_model(input_tensor,
						attention_mask=None,
						hidden_size=768,
						num_hidden_layers=12,
						num_attention_heads=12,
						intermediate_size=3072,
						intermediate_act_fn=gelu,
						hidden_dropout_prob=0.1,
						attention_probs_dropout_prob=0.1,
						initializer_range=0.02,
						do_return_all_layers=False,
						attention_fixed_size=None,
						dropout_name=None,
						structural_attentions="none",
						is_training=False,
						model_config={},
						from_mask=None,
						to_mask=None):
	"""Multi-headed, multi-layer Transformer from "Attention is All You Need".

	This is almost an exact implementation of the original Transformer encoder.

	See the original paper:
	https://arxiv.org/abs/1706.03762

	Also see:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
		attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
			seq_length], with 1 for positions that can be attended to and 0 in
			positions that should not be.
		hidden_size: int. Hidden size of the Transformer.
		num_hidden_layers: int. Number of layers (blocks) in the Transformer.
		num_attention_heads: int. Number of attention heads in the Transformer.
		intermediate_size: int. The size of the "intermediate" (a.k.a., feed
			forward) layer.
		intermediate_act_fn: function. The non-linear activation function to apply
			to the output of the intermediate/feed-forward layer.
		hidden_dropout_prob: float. Dropout probability for the hidden layers.
		attention_probs_dropout_prob: float. Dropout probability of the attention
			probabilities.
		initializer_range: float. Range of the initializer (stddev of truncated
			normal).
		do_return_all_layers: Whether to also return all layers or just the final
			layer.

	Returns:
		float Tensor of shape [batch_size, seq_length, hidden_size], the final
		hidden layer of the Transformer.

	Raises:
		ValueError: A Tensor shape or parameter is invalid.
	"""
	if not attention_fixed_size:
		if hidden_size % num_attention_heads != 0:
			raise ValueError(
					"The hidden size (%d) is not a multiple of the number of attention "
					"heads (%d)" % (hidden_size, num_attention_heads))

	attention_head_size = int(hidden_size / num_attention_heads)
	input_shape = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	input_width = input_shape[2]

	# The Transformer performs sum residuals on all layers so the input needs
	# to be the same as the hidden size.
	# if input_width != hidden_size:
	# 	raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
	# 									 (input_width, hidden_size))

	if input_width != hidden_size:
		input_tensor = bert_modules.dense_layer_2d(
		input_tensor, hidden_size, create_initializer(initializer_range),
		None, name="embedding_hidden_mapping_in")

		tf.logging.info("==apply embedding linear projection==")

	# We keep the representation as a 2D tensor to avoid re-shaping it back and
	# forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
	# the GPU/CPU but may not be free on the TPU, so we want to minimize them to
	# help the optimizer.
	prev_output = bert_utils.reshape_to_matrix(input_tensor)

	all_layer_outputs = []
	all_attention_scores = []
	all_value_outputs = []

	for layer_idx in range(num_hidden_layers):
		with tf.variable_scope("layer_%d" % layer_idx):
			layer_input = prev_output

			with tf.variable_scope("attention"):
				attention_heads = []
				with tf.variable_scope("self"):

					if dropout_name:
						attention_dropout_name = dropout_name + "/layer_%d/attention/self" % layer_idx
					else:
						attention_dropout_name = None
					if layer_idx in list(range(num_hidden_layers)):
						structural_attentions_args = structural_attentions
					else:
						structural_attentions_args = "none"

					[attention_scores,
					value_layer] = attention_layer(
							from_tensor=layer_input,
							to_tensor=layer_input,
							attention_mask=attention_mask,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length,
							attention_fixed_size=attention_fixed_size,
							dropout_name=attention_dropout_name,
							structural_attentions=structural_attentions_args,
							is_training=is_training)
					all_attention_scores.append(attention_scores)
					all_value_outputs.append(value_layer)

				# Run a linear projection of `hidden_size` then add a residual
				# with `layer_input`.
				with tf.variable_scope("output"):

					if dropout_name:
						output_dropout_name = dropout_name + "/layer_%d/attention/output" % layer_idx
					else:
						output_dropout_name = None

					value_output = tf.layers.dense(
							value_layer,
							hidden_size,
							kernel_initializer=bert_modules.create_initializer(initializer_range))
					
					attention_output = hard_attention(
								attention_scores=attention_scores,
								value_output=value_output,
								attention_mask=attention_mask, 
								batch_size=batch_size,
								from_seq_length=seq_length,
								to_seq_length=seq_length, 
								num_attention_heads=num_attention_heads,
								attention_head_size=attention_head_size,
								do_return_2d_tensor=True)

					attention_output = bert_modules.dropout(attention_output, hidden_dropout_prob, dropout_name=output_dropout_name)
					attention_output = bert_modules.layer_norm(attention_output + layer_input)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("intermediate"):
				intermediate_output = tf.layers.dense(
						attention_output,
						intermediate_size,
						activation=intermediate_act_fn,
						kernel_initializer=bert_modules.create_initializer(initializer_range))

			# Down-project back to `hidden_size` then add the residual.
			with tf.variable_scope("output"):

				if dropout_name:
					output_dropout_name = dropout_name + "/layer_%d/output" % layer_idx
				else:
					output_dropout_name = None

				layer_output = tf.layers.dense(
						intermediate_output,
						hidden_size,
						kernel_initializer=bert_modules.create_initializer(initializer_range))
				layer_output = bert_modules.dropout(layer_output, hidden_dropout_prob, dropout_name=output_dropout_name)
				layer_output = bert_modules.layer_norm(layer_output + attention_output)
				prev_output = layer_output
				all_layer_outputs.append(layer_output)

	if do_return_all_layers:
		final_outputs = []
		for layer_output in all_layer_outputs:
			final_output = bert_utils.reshape_from_matrix(layer_output, input_shape)
			final_outputs.append(final_output)
		return final_outputs, all_attention_scores, all_value_outputs
	else:
		final_output = bert_utils.reshape_from_matrix(prev_output, input_shape)
		return final_output, all_attention_scores, all_value_outputs

