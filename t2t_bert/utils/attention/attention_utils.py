import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
															 mode='FAN_AVG',
															 uniform=True,
															 dtype=tf.float32)

def mask_logits(inputs, mask, mask_value = -1e30):
	shapes = inputs.shape.as_list()
	mask = tf.cast(mask, tf.float32)
	return mask * inputs + mask_value * (1 - mask)

def query_context_alignment(query, context, 
				query_mask, context_mask,
				scope, reuse=None):
	
	with tf.variable_scope(scope+"/Context_to_Query_Attention_Layer", reuse=reuse):
		context_ = tf.transpose(context, [0,2,1])
		shape_lst = bert_utils.get_shape_list(query, expected_rank=3)
		
		hidden_dim = shape_lst[-1]

		attn_W = tf.get_variable("AttnW", dtype=tf.float32,
									shape=[hidden_dim, hidden_dim],
									initializer=initializer)

		weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]]) # batch x q_len x hidden_dim

		S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

		mask_q = tf.expand_dims(query_mask, 1) # batch x 1 x q_len 
		mask_c = tf.expand_dims(context_mask, 1) # batch x 1 x c_len

		# S_ = tf.nn.softmax(mask_logits(S, mask = mask_c))
		S_ = tf.exp(tf.nn.log_softmax(mask_logits(S, mask = mask_c))) # batch x q_len x c_len
		query_attn = tf.matmul(S_, context) # batch x q_len x hidden_dim

		S_T = tf.exp(tf.nn.log_softmax(mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q)))
		context_attn = tf.matmul(S_T, query)

		# query_attention_outputs = tf.concat([query, c2q, query-c2q, query*c2q], axis=-1)
		# query_attention_outputs *= tf.expand_dims(tf.cast(query_mask, tf.float32), -1)

		# context_attention_outputs = tf.concat([context, q2c, context-q2c, context*q2c], axis=-1)
		# context_attention_outputs *= tf.expand_dims(tf.cast(context_mask, tf.float32), -1)

		return [query_attn, context_attn]

def attention_bias_ignore_padding(memory_padding):
	"""Create an bias tensor to be added to attention logits.

	Args:
		memory_padding: a float `Tensor` with shape [batch, memory_length].

	Returns:
		a `Tensor` with shape [batch, 1, 1, memory_length].
		each dim corresponding to batch_size, num_heads, queries_len,
		memory_length
	"""
	ret = tf.multiply(memory_padding, -1e18)
	return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def _split_heads(x, num_heads):
	"""Split channels (dimension 2) into multiple heads,
		becomes dimension 1).
	Must ensure `x.shape[-1]` can be deviced by num_heads
	"""
	shape_lst = bert_utils.get_shape_list(x)
	depth = shape_lst[-1]
	batch = shape_lst[0]
	seq = shape_lst[1]
	# print(x.get_shape(), "===splitheads===")
	splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
		num_heads, depth // num_heads])
	return tf.transpose(splitted_x, [0, 2, 1, 3])

def _combine_heads(x):
	"""
	Args:
		x: A Tensor of shape `[batch, num_heads, seq_len, dim]`

	Returns:
		A Tensor of shape `[batch, seq_len, num_heads * dim]`
	"""
	t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
	num_heads, dim = t.get_shape()[-2:]
	return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])

def multihead_attention_texar(queries, 
				memory=None, 
				memory_attention_bias=None,
				num_heads=8, 
				num_units=None, 
				dropout_rate=0.0, 
				scope="multihead_attention"):
	if num_units is None:
		num_units = queries.get_shape()[-1]
	if num_units % num_heads != 0:
			raise ValueError("Value depth (%d) must be divisible by the"
							 "number of attention heads (%d)." % (\
							num_units, num_heads))
	if memory is None:
		Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
		K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
		V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
	else:
		Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
		K = tf.layers.dense(memory, num_units, use_bias=False, name='k')
		V = tf.layers.dense(memory, num_units, use_bias=False, name='v')

	Q_ = _split_heads(Q, num_heads)
	K_ = _split_heads(K, num_heads)
	V_ = _split_heads(V, num_heads)

	key_depth_per_head = num_units // num_heads
	Q_ *= tf.pow(tf.cast(key_depth_per_head, tf.float32), -0.5)

	logits = tf.matmul(Q_, K_, transpose_b=True)
	if memory_attention_bias is not None:
		logits += memory_attention_bias
	weights = tf.nn.softmax(logits, name="attention_weights")
	weights = tf.nn.dropout(weights, 1 - dropout_rate)
	outputs = tf.matmul(weights, V_)

	outputs = _combine_heads(outputs)
	outputs = tf.layers.dense(outputs, num_units,\
			use_bias=False, name='output_transform')
		#(batch_size, length_query, attention_depth)
	return outputs

def vector_attention(x, encode_dim, feature_dim, attention_dim, sequence_mask=None,
					 mask_zero=False, maxlen=None, epsilon=1e-8, seed=0,
					 scope_name="attention", reuse=False):
	"""
	:param x: [batchsize, s, feature_dim]
	:param encode_dim: dim of encoder output
	:param feature_dim: dim of x (for self-attention, x is the encoder output;
						for context-attention, x is the concat of encoder output and contextual info)
	:param sequence_length:
	:param mask_zero:
	:param maxlen:
	:param epsilon:
	:param seed:
	:param scope_name:
	:param reuse:
	:return: [batchsize, s, encode_dim]
	"""
	with tf.variable_scope(scope_name, reuse=reuse):
		# W1: [attention_dim, feature_dim]
		W1 = tf.get_variable("W1_%s" % scope_name,
							 initializer=tf.truncated_normal_initializer(
								 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
							 dtype=tf.float32,
							 shape=[attention_dim, feature_dim])
		# b1: [attention_dim]
		b1 = tf.get_variable("b1_%s" % scope_name,
							 initializer=tf.truncated_normal_initializer(
								 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
							 dtype=tf.float32,
							 shape=[attention_dim])
		# W2: [encode_dim, attention_dim]
		W2 = tf.get_variable("W2_%s" % scope_name,
							 initializer=tf.truncated_normal_initializer(
								 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
							 dtype=tf.float32,
							 shape=[encode_dim, attention_dim])
		# b2: [encode_dim]
		b2 = tf.get_variable("b2_%s" % scope_name,
							 initializer=tf.truncated_normal_initializer(
								 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
							 dtype=tf.float32,
							 shape=[encode_dim])
	# [batchsize, attention_dim, s]
	e = tf.nn.relu(
		tf.einsum("bsf,af->bas", x, W1) + \
		tf.expand_dims(tf.expand_dims(b1, axis=0), axis=-1))
	# [batchsize, s, encode_dim]
	e = tf.einsum("bas,ea->bse", e, W2) + \
		tf.expand_dims(tf.expand_dims(b2, axis=0), axis=0)
	# a = tf.exp(e)

	# apply mask after the exp. will be re-normalized next
	if mask_zero:
		# [batchsize, s, 1]
		# mask = tf.sequence_mask(sequence_length, maxlen)
		mask = tf.expand_dims(tf.cast(sequence_mask, tf.float32), axis=-1)
		a = mask_logits(e, mask, mask_value = -1e30)

	# in some cases especially in the early stages of training the sum may be almost zero
	a = tf.exp(tf.nn.log_softmax(a, axis=1))

	return a

def multihead_pooling(inputs, sequence_mask=None,num_units=None,
					 mask_zero=False,
					 num_heads=12,
					 scope_name="attention", reuse=False):

	with tf.variable_scope(scope_name, reuse=reuse):

		shape_lst = bert_utils.get_shape_list(inputs, expected_rank=3)

		if num_units:
			num_units = num_units
		else:
			num_units = shape_lst[-1]

		Q = tf.layers.dense(inputs, num_units, 
							activation = tf.nn.relu,
							name="generalized_pooling_q")

		K = tf.layers.dense(Q, num_units, 
							activation = tf.nn.relu,
							name="generalized_pooling_k")

		print(" ==K shape {}== ".format(K.get_shape()))

		# batch x seq x dim ---> (batch x seq x dim_ x num_heads
		
		shape_lst_K = bert_utils.get_shape_list(K)
		depth = shape_lst_K[-1]
		batch = shape_lst_K[0]
		seq = shape_lst_K[1]
		# print(x.get_shape(), "===splitheads===")
		K_ = tf.reshape(K, [batch, seq, \
		num_heads, depth // num_heads])

		print(" ==K_ shape {}== ".format(K_.get_shape()))

		# apply mask after the exp. will be re-normalized next
		if mask_zero:
			# batch x seq x 1 x 1
			mask = tf.expand_dims(tf.cast(sequence_mask, tf.float32), axis=-1)
			mask = tf.expand_dims(mask, axis=-1)
			K_ = mask_logits(K_, mask, mask_value = -1e30)

		weight = tf.exp(tf.nn.log_softmax(K_, axis=1)) # batch x seq x num_head x dim
		repres = tf.reduce_sum(K_ * weight, axis=1) # batch x num_head x dim

		return tf.reshape(repres, [shape_lst[0], num_units])



