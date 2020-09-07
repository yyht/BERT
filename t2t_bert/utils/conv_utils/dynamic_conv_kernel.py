
import tensorflow as tf
import numpy as np
from utils.bert import bert_utils
from utils.textcnn import position_utils

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

def mixed_dynamic_conv_attention_layer(from_tensor,
				to_tensor,
				attention_mask=None,
				from_mask=None,
				to_mask=None,
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
				kernel_size=10,
				strides=1,
				dilation_rate=1,
				scale_ratio=0.5):

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

	if scale_ratio < 1:
		num_attention_heads = tf.cast(num_attention_heads * ratio, dtype=tf.int32)
		tf.logging.info("==scale num_attention_heads==", num_attention_heads)

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
	to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

	# `query_layer` = [B*F, N*H]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=query_act,
			name="query",
			kernel_initializer=create_initializer(initializer_range))

	# `key_layer` = [B*T, N*H]
	key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=key_act,
			name="key",
			kernel_initializer=create_initializer(initializer_range))

	# `value_layer` = [B*T, N*H]
	value_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=value_act,
			name="value",
			kernel_initializer=create_initializer(initializer_range))

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

	if attention_mask is not None:
		# `attention_mask` = [B, 1, F, T]
		attention_mask = tf.expand_dims(attention_mask, axis=[1])

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		attention_scores += adder

	# Normalize the attention scores to probabilities.
	# `attention_probs` = [B, N, F, T]
	# attention_probs = tf.nn.softmax(attention_scores)
	attention_probs = tf.exp(tf.nn.log_softmax(attention_scores))

	# This is actually dropping out entire tokens to attend to, which might
	# seem a bit unusual, but is taken from the original Transformer paper.
	attention_probs = dropout(attention_probs, attention_probs_dropout_prob, dropout_name=dropout_name)

	# `value_layer` = [B, T, N, H]
	value_layer = tf.reshape(
			value_layer,
			[batch_size, to_seq_length, num_attention_heads, attention_head_size])

	# `value_layer` = [B, N, T, H]
	value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

	# `context_layer` = [B, N, F, H]
	context_layer = tf.matmul(attention_probs, value_layer)

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

	# dynamic-span-conv: key_layer
	# [batch_size, seq_length, num_attention_heads x width]
	if from_mask is not None:
		from_tensor_mask = from_mask
	else:
		from_tensor_mask = tf.squeeze(attention_mask[:, 0:1, :])
	
	from_tensor_mask = tf.expand_dims(from_tensor_mask, axis=-1)
	from_tensor_mask = tf.cast(from_tensor_mask, dtype=tf.float32)

	if len(from_shape) == 3:
		from_tensor *= from_tensor_mask
	else:
		from_tensor = tf.reshape(
				from_tensor,
				[batch_size * from_seq_length, num_attention_heads * attention_head_size])
		from_tensor *= from_tensor_mask
	conv_key_layer = gated_conv1d_op(from_tensor, 
								filters=num_attention_heads * attention_head_size, 
								kernel_size=kernel_size, 
								padding="SAME", 
								activation=None, 
								strides=strides, 
								reuse=True, 
								name="glu_conv", 
								kernel_initializer=None,
								dilation_rate=dilation_rate,
								is_training=True)
	conv_key_layer *= from_tensor_mask

	# [batch_size, num_attention_heads, seq_length, width]
	conv_key_layer = tf.reshape(conv_key_layer, 
												[batch_size,
												from_seq_length,
												num_attention_heads,
												attention_head_size])
	conv_key_layer = tf.transpose(conv_key_layer, [0, 2, 1, 3])

	# dynamic-kernel-generator
	dynamic_kernel_generator = conv_key_layer * query_layer

	# kernel-project-layer
	dynamic_kernel = tf.get_variable(
				"dynamic_kernel",
				shape=[num_attention_heads, kernel_size, attention_head_size],
				initializer=create_initializer(initializer_range))

	# [batch_size, num_attention_heads, seq_length, kernel_size]
	dynamic_conv_kernel = tf.einsum("abcd,bfd->abcf", 
									dynamic_kernel_generator, 
									dynamic_kernel)
	# [batch_size, num_attention_heads, seq_length, kernel_size]
	normalized_dynamic_kernel = tf.exp(tf.log_softmax(dynamic_conv_kernel, axis=-1))
	normalized_dynamic_kernel = dropout(normalized_dynamic_kernel, 
										attention_probs_dropout_prob, 
										dropout_name=dropout_name+"_conv")
	# [1, 1, seq_length, seq_length]
	band_matrix = tf.matrix_band_part(tf.ones((
							from_seq_length+kernel_size-1, 
							from_seq_length+kernel_size-1)), 
							0, kernel_size-1)

	# [to_seq_length, to_seq_length+(kernel_size-1)]
	final_bank_matrix = band_matrix[:-(kernel_size-1), :]

	indices = tf.contrib.framework.argsort(final_bank_matrix, direction='DESCENDING')
	# [to_seq_length, kernel_size]
	indices = indices[:, :kernel_size]
	indices = tf.reverse(indices, axis=[-1])

	# padded_value_layer: [batch_size, num_attention_heads, seq_length+kernel_size-1, attention_head_size]
	padded_value_layer = tf.pad(value_layer, 
							[[0, 0], 
							[0, 0], 
							[int((kernel_size-1)/2) ,int((kernel_size-1)/2)], 
							[0,0]])

	# [1, to_seq_length*kernel_size]
	indices = tf.expand_dims(tf.reshape(indices, [-1]), axis=0)
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
	conv_output = tf.einsum("abcd,abcde->abce", normalized_dynamic_kernel, conv_span_output)
	conv_output = tf.transpose(conv_output, [0, 2, 1, 3])
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

	final_context_layer = tf.concat([context_layer, conv_output_layer], axis=-1)
	return final_context_layer, final_context_layer, value_layer