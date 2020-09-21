
import tensorflow as tf
import numpy as np

from utils.textcnn import light_conv_utils
from utils.bert import bert_utils

from utils.bert import dropout_utils

stable_dropout = dropout_utils.ReuseDropout()

def dropout(input_tensor, dropout_prob, dropout_name=None):
	"""Perform dropout.

	Args:
		input_tensor: float Tensor.
		dropout_prob: Python float. The probability of dropping out a value (NOT of
			*keeping* a dimension as in `tf.nn.dropout`).

	Returns:
		A version of `input_tensor` with dropout applied.
	"""
	if dropout_prob is None or dropout_prob == 0.0:
		return tf.identity(input_tensor)
	if dropout_name:
		output = stable_dropout.dropout(input_tensor, dropout_prob, dropout_name)
	else:
		output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output

def dynamic_conv_layer(
				from_tensor,
				from_mask=None,
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
				attention_fixed_size=None,
				dropout_name=None,
				structural_attentions="none",
				is_training=False,
				kernel_size=10,
				strides=1,
				dilation_rate=1,
				scale_ratio=0.5,
				is_casual=False):
	def transpose_for_scores(input_tensor, batch_size, 
														num_attention_heads,
														seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])

		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor

	from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])

	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None):
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

	if scale_ratio < 1:
		num_attention_heads = int(num_attention_heads * ratio)
		tf.logging.info("==scale num_attention_heads==", num_attention_heads)

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)

	# `value_layer` = [B*T, N*H]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=value_act,
			name="query",
			kernel_initializer=create_initializer(initializer_range))

	# `value_layer` = [B*T, N*H]
	value_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=value_act,
			name="value",
			kernel_initializer=create_initializer(initializer_range))

	# `query_layer` = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size,
									 num_attention_heads, 
									 from_seq_length,
									 attention_head_size)

	# `query_layer` = [B, N, F, H]
	value_layer = transpose_for_scores(value_layer, batch_size,
									 num_attention_heads, 
									 from_seq_length,
									 attention_head_size)

	# dynamic-span-conv: key_layer
	# [batch_size, seq_length, num_attention_heads x width]
	
	from_tensor_mask = tf.expand_dims(from_mask, axis=-1)
	from_tensor_mask = tf.cast(from_tensor_mask, dtype=tf.float32)

	if len(from_shape) == 3:
		from_tensor *= from_tensor_mask
	else:
		from_tensor = tf.reshape(
				from_tensor,
				[batch_size, from_seq_length, num_attention_heads * attention_head_size])

	from_tensor *= from_tensor_mask

	if is_casual:
		left_pad = dilation_rate * (kernel_size - 1)
		dynamic_span_inputs = tf.pad(from_tensor, [[0, 0], [left_pad, 0], [0, 0]])
		padding = 'VALID'
		tf.logging.info("==casual valid padding==")
	else:
		dynamic_span_inputs = from_tensor
		padding = 'SAME'

	conv_key_layer = light_conv_utils.gated_conv1d_op(dynamic_span_inputs, 
								filters=num_attention_heads * attention_head_size, 
								kernel_size=kernel_size, 
								padding=padding, 
								activation=None, 
								strides=strides, 
								reuse=None, 
								name="glu_conv", 
								kernel_initializer=None,
								dilation_rate=dilation_rate,
								is_training=is_training)
	conv_key_layer *= from_tensor_mask

	# [batch_size, num_attention_heads, seq_length, width]
	conv_key_layer = tf.reshape(conv_key_layer, [batch_size,
												from_seq_length,
												num_attention_heads,
												attention_head_size])
	conv_key_layer = tf.transpose(conv_key_layer, [0, 2, 1, 3])

	# dynamic-kernel-generator
	# [batch_size, num_attention_heads, seq_length, attention_head_size]
	# dynamic_kernel_generator = tf.identity(conv_key_layer)
	dynamic_kernel_generator = conv_key_layer * query_layer

	# kernel-project-layer
	dynamic_kernel = tf.get_variable(
				"dynamic_kernel",
				shape=[num_attention_heads, kernel_size, attention_head_size],
				initializer=create_initializer(initializer_range))

	# [batch_size, num_attention_heads, from_seq_length, attention_head_size]
	# [num_attention_heads, kernel_size, attention_head_size]
	dynamic_conv_kernel = tf.einsum("abcd,bfd->abcf", 
									dynamic_kernel_generator, 
									dynamic_kernel)

	# [batch_size, num_attention_heads, seq_length, kernel_size]
	normalized_dynamic_kernel = tf.exp(tf.nn.log_softmax(dynamic_conv_kernel, axis=-1))
	normalized_dynamic_kernel = dropout(normalized_dynamic_kernel, 
										attention_probs_dropout_prob, 
										dropout_name=dropout_name+"_conv")
	# [1, from_seq_length+kernel_size-1]
	indices_i = tf.range(from_seq_length+kernel_size-1, delta=1)
	indices = tf.reverse(indices_i[0:kernel_size], axis=[-1])
	indices = tf.expand_dims(indices, axis=0)

	batch_one = tf.ones((from_seq_length, 1), dtype=indices.dtype)
	batch_index = tf.einsum("ab,bc->ac", batch_one, indices)

	incremental_index = tf.transpose(tf.expand_dims(indices_i[:from_seq_length], axis=0))
	indices += incremental_index
	
	indices = tf.reshape(indices, [-1])

	# padded_value_layer: [batch_size, num_attention_heads, seq_length+kernel_size-1, attention_head_size]
	if is_casual:
		padded_value_layer = tf.pad(value_layer, 
								[[0, 0], 
								[0, 0]，
								[(kernel_size - 1), 0], 
								[0, 0]])
		tf.logging.info("==casual valid padding==")
	else:
		padded_value_layer = tf.pad(value_layer, 
								[[0, 0], 
								[0, 0], 
								[int((kernel_size-1)/2), int((kernel_size-1)/2)], 
								[0,0]])

	# [1, to_seq_length*kernel_size]
	conv_span_output = bert_utils.gather_indexes(padded_value_layer, indices)
	conv_span_output = tf.reshape(conv_span_output, 
								[batch_size, 
								num_attention_heads,
								from_seq_length,
								kernel_size,
								attention_head_size
								])
	
	# dynamic_conv_kernel:[batch_size, num_attention_heads, seq_length, kernel_size]
	# conv_span_output:    [batch_size, num_attention_heads, seq_length, kernel_size, attention_head_size]
	conv_output = tf.einsum("abcd,abcde->abce", 
							normalized_dynamic_kernel, 
							conv_span_output)
	conv_output = tf.transpose(conv_output, [0, 2, 1, 3])
	conv_output *= tf.expand_dims(from_tensor_mask, axis=-1)
	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*V]
		conv_output_layer = tf.reshape(
				conv_output,
				[batch_size * from_seq_length, num_attention_heads * attention_head_size])
	else:
		# `context_layer` = [B, F, N*V]
		conv_output_layer = tf.reshape(
				conv_output,
				[batch_size, from_seq_length, num_attention_heads * attention_head_size])

	return conv_output_layer


def dynamic_dgcnn(x, input_mask,
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
			attention_fixed_size=None,
			dropout_name=None,
			structural_attentions="none",
			scale_ratio=0.5,
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
			padding='SAME',
			layer_wise_pos=False,
			):

	print(num_filters, '===num_filters===')

	# input_mask: batch_size, seq

	# initializer = tf.glorot_uniform_initializer()
	initializer = create_initializer(initializer_range=0.02)

	input_mask = tf.cast(input_mask, dtype=tf.float32)
	outter_input_mask = tf.expand_dims(input_mask, axis=-1)

	padding_type = padding

	if is_training:
		attention_probs_dropout_prob = attention_probs_dropout_prob
	else:
		attention_probs_dropout_prob = 0.0

	if is_casual:
		left_pad = dilation_rates[0] * (kernel_sizes[0] - 1)
		inputs = tf.pad(x, [[0, 0], [left_pad, 0], [0, 0]])
		padding = 'VALID'
		tf.logging.info("==casual valid padding==")
	else:
		inputs = x
		# left_pad = int(dilation_rates[0] * (kernel_sizes[0] - 1) / 2)
		# right_pad = int(dilation_rates[0] * (kernel_sizes[0] - 1) / 2)
		# print(left_pad, right_pad, '===projection===')
		# inputs = tf.pad(x, [[0, 0], [left_pad, right_pad], [0, 0]])
		# padding = 'VALID'
		padding = 'SAME'

	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0

	if layer_wise_pos:
		inputs = position_utils.add_timing_signal_1d(inputs)
		tf.logging.info("==layer-wise position encoding==")

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
		if padding_type == 'SAME':
			inputs *= outter_input_mask
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
			
			tf.logging.info("==kernel_size:%s, num_filter:%s, stride:%s, dilation_rate:%s==", str(kernel_size), 
										str(num_filter), str(stride), str(dilation_rate))
			if layer_wise_pos:
				inputs = position_utils.add_timing_signal_1d(inputs)
				tf.logging.info("==layer-wise position encoding==")
			gatedcnn_outputs = dynamic_conv_layer(
								inputs,
								from_mask=input_mask,
								num_attention_heads=num_attention_heads,
								size_per_head=size_per_head,
								query_act=None,
								key_act=None,
								value_act=None,
								attention_probs_dropout_prob=attention_probs_dropout_prob,
								initializer_range=0.02,
								do_return_2d_tensor=False,
								batch_size=None,
								from_seq_length=None,
								attention_fixed_size=None,
								dropout_name=None,
								structural_attentions="none",
								is_training=is_training,
								kernel_size=kernel_size,
								strides=stride,
								dilation_rate=dilation_rate,
								scale_ratio=1.0,
								is_casual=is_casual)

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

			if padding_type == 'SAME':
				inputs *= outter_input_mask
			residual_inputs = inputs
	
	return inputs