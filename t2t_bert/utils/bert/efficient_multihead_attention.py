import tensorflow as tf
import numpy as np

from utils.bert import bert_utils
from utils.bert import albert_modules

def efficient_attention_layer(from_tensor,
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
										attention_fixed_size=None):
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
		tf.logging.info("==apply attention_fixed_size==", str(attention_head_size))
	else:
		attention_head_size = size_per_head
		tf.logging.info("==apply attention_original_size==", str(attention_head_size))

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
	to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

	# `query_layer` = [B*F, N*H]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=query_act,
			name="query",
			kernel_initializer=albert_modules.create_initializer(initializer_range))

	# `key_layer` = [B*T, N*H]
	key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=key_act,
			name="key",
			kernel_initializer=albert_modules.create_initializer(initializer_range))

	# `value_layer` = [B*T, N*H]
	value_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=value_act,
			name="value",
			kernel_initializer=albert_modules.create_initializer(initializer_range))

	# softmax(QK^T/sqrt(4))V
	#softmax(Q)softmax(K)^TV

	# `query_layer` = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size,
									 num_attention_heads, from_seq_length,
									 attention_head_size)

	# `key_layer` = [B, N, T, H]
	key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
									to_seq_length, attention_head_size)

	# `value_layer` = [B, N, T, H]
	value_layer = transpose_for_scores(value_layer, batch_size, num_attention_heads,
									to_seq_length, attention_head_size)

	# Take the dot product between "query" and "key" to get the raw
	# attention scores.
	# `attention_scores` = [B, N, H, H]<---[B, N, T, H] x [B, N, T, H]
	# key_mask = [B, T, 1, 1]

	attention_mask = tf.cast(tf.expand_dims(attention_mask[:, 0:1, :], axis=[2]), tf.float32)
	attention_mask = tf.cast(tf.expand_dims(attention_mask, axis=[3]), tf.float32)
	# key_mask = [B, 1, T, 1]
	attention_mask = tf.reshape(attention_mask, [batch_size, 1, to_seq_length, 1])
	adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
	attention_scores = tf.nn.log_softmax(key_layer+adder, axis=2)
	attention_probs = tf.exp(attention_scores)
	attention_probs = albert_modules.dropout(attention_probs, attention_probs_dropout_prob)
	
	key_value_scores = tf.matmul(attention_probs, value_layer, transpose_a=True)

	# This is actually dropping out entire tokens to attend to, which might
	# seem a bit unusual, but is taken from the original Transformer paper.
	# [B, N, F, H] x [B, N, H, H]--->[B, N, F, H]
	context_layer = tf.matmul(tf.exp(tf.nn.log_softmax(query_layer, axis=-1)), key_value_scores)

	# `context_layer` = [B, F, N, H]
	context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		context_layer = tf.reshape(
				context_layer,
				[batch_size * from_seq_length, num_attention_heads * attention_head_size])
	else:
		# `context_layer` = [B, F, N*V]
		context_layer = tf.reshape(
				context_layer,
				[batch_size, from_seq_length, num_attention_heads * attention_head_size])

	return context_layer, attention_scores