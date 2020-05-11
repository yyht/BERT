import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers
from utils.bert import bert_utils
import tensorflow.contrib.layers as layers
from utils.dsmm.tf_common import nn_module 

def text_cnn(in_val, filter_sizes, scope, 
	embed_size, num_filters, max_pool_size=2):
	print(in_val.get_shape(), "===in_val shape===")
	in_val = tf.expand_dims(in_val, axis=-1)
	conved_concat = []

	in_val_shape = input_shape = bert_utils.get_shape_list(in_val, expected_rank=[4])
	sequence_length = in_val_shape[1]
	for i, filter_size in enumerate(filter_sizes):
		with tf.variable_scope(scope+'_conv-maxpool-%s' % filter_size):

			filter_shape = [filter_size, embed_size, 1, num_filters]
			W = tf.get_variable("W", 
								shape=filter_shape,
								dtype=tf.float32,
								initializer=tf.truncated_normal_initializer(stddev=0.1)
								)
			b = tf.get_variable("b", 
								shape=[num_filters],
								dtype=tf.float32,
								initializer=tf.constant_initializer(0.1)
								)
			conv = tf.nn.conv2d(in_val, W, strides=[1, 1, 1, 1], 
								padding='VALID', name='conv')
			print(conv.get_shape(), "====conv shape====")
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
			# pooled = tf.reduce_max(h, axis=1)

			pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")

			print(pooled.get_shape(), filter_size, "==filter_size==")
			conved_concat.append(pooled)

	cnn_output = tf.concat(conved_concat, -1)
	print(cnn_output.get_shape(), "==cnn_output==")
	last_dim = len(filter_sizes) * num_filters
	cnn_output = tf.reshape(cnn_output, [-1, last_dim])

	print(cnn_output.get_shape(), "=====text cnn output=====")
	return cnn_output

def text_cnn_v1(in_val, filter_sizes, scope, 
	embed_size, num_filters, max_pool_size=2, input_mask=None):
	print(in_val.get_shape(), "===in_val shape===")
	in_val = tf.expand_dims(in_val, axis=-1)
	conved_concat = []

	# input_mask = tf.cast(tf.expand_dims(input_mask, axis=[2]), tf.float32)
	# input_mask = tf.expand_dims(input_mask, axis=[3])

	in_val_shape = input_shape = bert_utils.get_shape_list(in_val, expected_rank=[4])
	sequence_length = in_val_shape[1]
	
	for i, filter_size in enumerate(filter_sizes):
		with tf.variable_scope(scope+'_conv-maxpool-%s' % filter_size):

			filter_shape = [filter_size, embed_size, 1, num_filters]
			W = tf.get_variable("W", 
								shape=filter_shape,
								dtype=tf.float32,
								initializer=tf.truncated_normal_initializer(stddev=0.1)
								)
			b = tf.get_variable("b", 
								shape=[num_filters],
								dtype=tf.float32,
								initializer=tf.constant_initializer(0.1)
								)
			conv = tf.nn.conv2d(in_val, W, strides=[1, 1, 1, 1], 
								padding='VALID', name='conv')
			print(conv.get_shape(), "====conv shape====")
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
			# pooled = tf.reduce_mean(h, axis=1)

			pooled = tf.nn.avg_pool(h, 
									ksize=[1, sequence_length - filter_size + 1, 1, 1],
									strides=[1, 1, 1, 1], 
									padding='VALID')

			print(pooled.get_shape(), filter_size, "==filter_size==")
			conved_concat.append(pooled)

	cnn_output = tf.concat(conved_concat, -1)
	print(cnn_output.get_shape(), "==cnn_output==")
	last_dim = len(filter_sizes) * num_filters
	cnn_output = tf.reshape(cnn_output, [-1, last_dim])

	print(cnn_output.get_shape(), "=====text cnn output=====")
	return cnn_output

def shape_list(x):
	"""Return list of dims, statically where possible."""

	x = tf.convert_to_tensor(x)

	# If unknown rank, return dynamic shape
	if x.get_shape().dims is None:
		return tf.shape(x)

	static = x.get_shape().as_list()
	shape = tf.shape(x)

	ret = []
	for i in range(len(static)):
		dim = static[i]
		if dim is None:
			dim = shape[i]
		ret.append(dim)
	return ret

def get_timing_signal_1d(length,
						 channels,
						 min_timescale=1.0,
						 max_timescale=1.0e4,
						 start_index=0):
	"""Gets a bunch of sinusoids of different frequencies.

	  Each channel of the input Tensor is incremented by a sinusoid of a different
	  frequency and phase.

	  This allows attention to learn to use absolute and relative positions.
	  Timing signals should be added to some precursors of both the query and the
	  memory inputs to attention.

	  The use of relative position is possible because sin(x+y) and cos(x+y) can be
	  expressed in terms of y, sin(x) and cos(x).

	  In particular, we use a geometric sequence of timescales starting with
	  min_timescale and ending with max_timescale.  The number of different
	  timescales is equal to channels / 2. For each timescale, we
	  generate the two sinusoidal signals sin(timestep/timescale) and
	  cos(timestep/timescale).  All of these sinusoids are concatenated in
	  the channels dimension.

	  Args:
		length: scalar, length of timing signal sequence.
		channels: scalar, size of timing embeddings to create. The number of
			different timescales is equal to channels / 2.
		min_timescale: a float
		max_timescale: a float
		start_index: index of first position

	  Returns:
		a Tensor of timing signals [1, length, channels]
	"""
	import math
	position = tf.to_float(tf.range(length) + start_index)
	num_timescales = channels // 2
	log_timescale_increment = (
	  math.log(float(max_timescale) / float(min_timescale)) /
	  (tf.to_float(num_timescales) - 1))
	inv_timescales = min_timescale * tf.exp(
	  tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
	scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
	signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
	signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
	signal = tf.reshape(signal, [1, length, channels])
	return signal

def multihead_attention(queries, 
						keys,
						queries_mask,
						keys_mask,
						num_units=None, 
						num_heads=8, 
						dropout_rate=0,
						is_training=True,
						causality=False,
						scope="multihead_attention", 
						reuse=None):
	'''Applies multihead attention.
	
	Args:
		queries: A 3d tensor with shape of [N, T_q, C_q].
		keys: A 3d tensor with shape of [N, T_k, C_k].
		num_units: A scalar. Attention size.
		dropout_rate: A floating point number.
		is_training: Boolean. Controller of mechanism for dropout.
		causality: Boolean. If true, units that reference the future are masked. 
		num_heads: An int. Number of heads.
		scope: Optional scope for `variable_scope`.
		reuse: Boolean, whether to reuse the weights of a previous layer
			by the same name.
			
	Returns
		A 3d tensor with shape of (N, T_q, C)   
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Set the fall back option for num_units
		if num_units is None:
			num_units = queries.get_shape().as_list[-1]

		queries_mask = tf.expand_dims(queries_mask, -1)
		queries *= tf.cast(queries_mask, tf.float32)

		keys_mask = tf.expand_dims(keys_mask, -1)
		keys *= tf.cast(keys_mask, tf.float32)
		
		# Linear projections
		Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
		K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		
		# Split and concat
		Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
		K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
		V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

		# Multiplication
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
		
		# Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
		
		# Key Masking
		key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
		key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
		key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
		
		paddings = tf.ones_like(outputs)*(-2**32+1)
		outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

		# Causality = Future blinding
		if causality:
			diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
			tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

			paddings = tf.ones_like(masks)*(-2**32+1)
			outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

		# Activation
		outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
		 
		# Query Masking
		query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
		query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
		query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
		outputs *= query_masks # broadcasting. (N, T_q, C)
			
		# Dropouts
		outputs = tf.nn.dropout(outputs, dropout_rate)
					 
		# Weighted sum
		outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
		
		# Restore shape
		outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
					
		# Residual connection
		outputs += queries
					
		# Normalize
		outputs = tf.contrib.layers.layer_norm(outputs)

		# outputs = normalize(outputs) # (N, T_q, C)
		outputs = tf.cast(outputs, dtype=tf.float32)

	return outputs

def feedforward(inputs, 
				num_units=[2048, 512],
				scope="multihead_attention", 
				reuse=None):
	'''Point-wise feed forward net.
	
	Args:
		inputs: A 3d tensor with shape of [N, T, C].
		num_units: A list of two integers.
		scope: Optional scope for `variable_scope`.
		reuse: Boolean, whether to reuse the weights of a previous layer
			by the same name.
			
	Returns:
		A 3d tensor with the same shape and dtype as inputs
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Inner layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
							"activation": tf.nn.relu, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		
		# Readinner layer
		params = {"inputs": outputs, "filters": num_units[0], "kernel_size": 5,
							"activation": tf.nn.relu, "use_bias": True, "padding":"same"}
		outputs = tf.layers.conv1d(**params)


		# Readout layer
		params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
							"activation": None, "use_bias": True, "padding":"same"}
		outputs = tf.layers.conv1d(**params)
		# Residual connection
		outputs += inputs
		
		# Normalize
		# outputs = normalize(outputs)
		outputs = tf.contrib.layers.layer_norm(outputs)
		outputs = tf.cast(outputs, dtype=tf.float32)
	
	return outputs

def task_specific_attention(inputs, output_size, input_mask,
							initializer=layers.xavier_initializer(),
							activation_fn=tf.tanh, scope=None, reuse=None):
	"""
	Performs task-specific attention reduction, using learned
	attention context vector (constant within task of interest).
	self-attentive sentence embedding

	Args:
		inputs: Tensor of shape [batch_size, units, input_size]
			`input_size` must be static (known)
			`units` axis will be attended over (reduced from output)
			`batch_size` will be preserved
		output_size: Size of output's inner (feature) dimension

	Returns:
		outputs: Tensor of shape [batch_size, output_dim].
	"""
	assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

	with tf.variable_scope(scope + '_attention', reuse=reuse) as scope:
		print("--------------using self attention----------------")
		attention_context_vector = tf.get_variable(name='attention_context_vector',
												   shape=[output_size],
												   initializer=initializer,
												   dtype=tf.float32)
		input_projection = layers.fully_connected(inputs, output_size,
												  activation_fn=activation_fn,
												  scope=scope) # batch x max_len x output_size

		vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2) # batch x max_len
		input_mask = tf.cast(input_mask, tf.float32)
		attention_weights = tf.nn.softmax(qanet_layers.mask_logits(vector_attn, mask = input_mask))
		attention_weights = tf.expand_dims(attention_weights, -1)
		# vector_attn_max = tf.reduce_max(qanet_layers.mask_logits(vector_attn, extend_mask), axis=1)
					
		# attention_weights = tf.exp(vector_attn-vector_attn_max) * tf.cast(extend_mask, tf.float32) # batch x max_len x 1
		# attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=1, keep_dims=True) # batch x max_len x 1
		
		weighted_projection = tf.multiply(input_projection, attention_weights)

		outputs = tf.reduce_sum(weighted_projection, axis=1)

		return outputs

def self_attn(enc, mask, scope, 
				dropout, config, reuse):

	length = shape_list(enc)[1]
	channels = shape_list(enc)[2]
						 
	position_embed = get_timing_signal_1d(length, channels)

	enc += position_embed
	for i in range(config.num_blocks):
		with tf.variable_scope(scope+"_encoder_num_blocks_{}".format(i), reuse=reuse):
			### Multihead Attention
			enc = multihead_attention(queries=enc, 
									keys=enc,
									queries_mask=mask,
									keys_mask=mask,
									scope="multihead_attention_{}".format(i),
									num_units=config.hidden_units, 
									num_heads=config.num_heads, 
									dropout_rate=1 - dropout,
									is_training=False,
									causality=False,
									reuse=reuse)
			
			### Feed Forward
			enc = feedforward(enc, 
							num_units=[config.hidden_units, 
										config.hidden_units],
							scope="ffn_{}".format(i),
							reuse=reuse)

	mask_ = tf.expand_dims(mask, -1)
	mask_ = tf.cast(mask_, tf.float32)

	enc *= mask_

	return enc

def gated_cnn(x, num_layers=2, num_filters=8, filter_sizes=[2, 3], 
			bn=False, training=False,
			timedistributed=False, scope_name="textcnn", 
			reuse=False, activation=tf.nn.relu,
			gated_conv=False, residual=False):
	if gated_conv:
		if residual:
			conv_op = nn_module.residual_gated_conv1d_op
			print("==apply gated_conv residual==")
		else:
			conv_op = nn_module.gated_conv1d_op
			print("==apply conv_op==")
	else:
		conv_op = tf.layers.conv1d
		print("==apply conv1d==")

	conv_blocks = []
	for i, filter_size in enumerate(filter_sizes):
		filter_scope_name = "filter_size_%s"%(str(filter_size))
		res_input = x
		for j in range(num_layers):
			layer_scope_name = "%s_layer_%s"%(str(scope_name), str(j))
			with tf.variable_scope(layer_scope_name, reuse=reuse):
				h = conv_op(
					inputs=res_input,
					filters=num_filters,
					kernel_size=filter_size,
					padding="same",
					activation=None,
					strides=1,
					reuse=reuse,
					name=filter_scope_name,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
				bias = tf.get_variable("b-%s" % filter_size, 
										[num_filters],
										initializer=tf.constant_initializer(0.1))  # ADD 2017-06-09
				h = tf.nn.relu(tf.nn.bias_add(h, bias), "relu")  # shape
				if bn:
					h = tf.contrib.layers.batch_norm(h, is_training=training, scope='bn')
				res_input = h
		conv_blocks.append(h)

	if len(conv_blocks) > 1:
		z = tf.concat(conv_blocks, axis=-1)
	else:
		z = conv_blocks[0]

	return z

def resnet_cnn(x, num_layers=2, num_filters=8, filter_sizes=[2, 3], 
			bn=False, training=False,
			timedistributed=False, scope_name="textcnn", 
			reuse=False, activation=tf.nn.relu,
			gated_conv=False, residual=False):
	if gated_conv:
		if residual:
			conv_op = nn_module.residual_gated_conv1d_op
			print("==apply gated_conv residual==")
		else:
			conv_op = nn_module.gated_conv1d_op
			print("==apply conv_op==")
	else:
		conv_op = tf.layers.conv1d
		print("==apply conv1d==")

	conv_blocks = []
	for i, filter_size in enumerate(filter_sizes):
		filter_scope_name = "filter_size_%s"%(str(filter_size))
		res_input = x
		for j in range(num_layers):
			layer_scope_name = "%s_layer_%s"%(str(scope_name), str(j))
			with tf.variable_scope(layer_scope_name, reuse=reuse):
				h = conv_op(
					inputs=res_input,
					filters=num_filters,
					kernel_size=filter_size,
					padding="same",
					activation=None,
					strides=1,
					reuse=reuse,
					name=filter_scope_name,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
				if bn and j < num_layers - 1:
					bn_scope = filter_scope_name +"_bn"
					h = tf.contrib.layers.batch_norm(h, is_training=training, scope=bn_scope)
				if j < num_layers - 1:
					h = tf.nn.relu(h)
				res_input = h
		# h = tf.nn.relu(h + x)
		h = tf.tanh(h)
		# h = tf.nn.relu(h)
		conv_blocks.append(h)

	if len(conv_blocks) > 1:
		z = tf.concat(conv_blocks, axis=-1)
	else:
		z = conv_blocks[0]

	return z

def cnn_multiple_layers(in_val, filter_sizes, scope, 
	embed_size, num_filters, max_pool_size=2,
	input_mask=None, is_training_flag=True):
	# 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
	# you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable

	print(in_val.get_shape(), "===in_val shape===")
	sentence_embeddings_expanded = tf.expand_dims(in_val, axis=-1)
	initializer = tf.truncated_normal_initializer(stddev=0.1)
	sequence_length = sentence_embeddings_expanded.get_shape()[1]
	pooled_outputs = []
	print("sentence_embeddings_expanded:", sentence_embeddings_expanded)
	for i, filter_size in enumerate(filter_sizes):
		with tf.variable_scope(scope+'/'+'cnn_multiple_layers' + "convolution-pooling-%s" % filter_size, 
							   reuse=tf.AUTO_REUSE):
			# 1) CNN->BN->relu
			filter1 = tf.get_variable("filter1-%s" % filter_size,[filter_size, embed_size, 1, num_filters],initializer=initializer)
			regoin_embedding = tf.nn.conv2d(sentence_embeddings_expanded, filter1, strides=[1, 1, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
			conv = tf.contrib.layers.batch_norm(regoin_embedding, is_training=is_training_flag, scope='cnn1')
			print(i, filter_size, "conv1:", conv)
			b = tf.get_variable("b-%s" % filter_size, [num_filters])  # ADD 2017-06-09
			h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

			# 2) CNN->BN->relu
#             h = tf.squeeze(h, axis=2) # batch x (sequence_length-1) x num_filters
#             h = tf.expand_dims(h, axis=-1) # batch x (sequence_length-1) x num_filters x 1
#             h = tf.reshape(h, [-1, sequence_length, num_filters,1])  # shape:[batch_size,sequence_length,num_filters,1]
			print(i, filter_size, "h:", h) # batch x (sequence_length-1) x 1 x num_filters
			# Layer2:CONV-RELU
			filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, 1, num_filters, num_filters],initializer=initializer)
			conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="SAME",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
			conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training_flag, scope='cnn2')
			print(i, "conv2:", conv2)
			b2 = tf.get_variable("b2-%s" % filter_size, [num_filters])  # ADD 2017-06-09
			h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
			print(i, filter_size, "h:", h)

			# 3. Max-pooling
#             pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1,sequence_length, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
			pooling_avg = tf.reduce_mean(h, axis=1) #[batch_size,num_filters]
			print(i, filter_size, "pooling:", pooling_avg)
			# pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
			pooled_outputs.append(pooling_avg)  # h:[batch_size,sequence_length,1,num_filters]

	cnn_output = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
	last_dim = len(filter_sizes) * num_filters
	cnn_output = tf.reshape(cnn_output, [-1, last_dim])
	print("h.concat:", cnn_output)
	return cnn_output
