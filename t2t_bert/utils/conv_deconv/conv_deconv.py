import tensorflow as tf
from utils.textcnn import conv1d_transpose as my_conv1d_transpose


def regularization(X, opt, is_train, prefix= '', is_reuse= None):
	if '_X' not in prefix and '_H_dec' not in prefix:
		if opt.batch_norm:
			X = layers.batch_norm(X, decay=0.9, center=True, scale=True, is_training=is_train, scope=prefix+'_bn', reuse = is_reuse)
		X = tf.nn.relu(X)
	X = X if not opt.cnn_layer_dropout else layers.dropout(X, keep_prob = opt.dropout_ratio, scope=prefix + '_dropout')

	return X

conv_acf = tf.nn.tanh # tf.nn.relu

def conv_model(X, opt, prefix = '', is_reuse= None, is_train = True):  # 2layers
	#XX = tf.reshape(X, [-1, , 28, 1])
	#X shape: batchsize L emb 1
	if opt.reuse_cnn:
		biasInit = opt.cnn_b
		weightInit = opt.cnn_W
	else:
		biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
		weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

	X = regularization(X, opt,  
					prefix=prefix+'reg_X', 
					is_reuse=is_reuse, 
					is_train=is_train)
	H1 = tf.layers.conv2d(X,  
					num_outputs=opt.filter_size,  
					kernel_size=[opt.filter_shape, opt.embed_size], 
					stride=[opt.stride[0],1],  
					weights_initializer=weightInit, 
					biases_initializer=biasInit, 
					activation_fn=None, 
					padding='VALID', 
					scope=prefix + 'H1', 
					reuse=is_reuse)  # batch L-3 1 Filtersize

	H1 = regularization(H1, opt, 
					prefix=prefix + 'reg_H1', 
					is_reuse=is_reuse, 
					is_train=is_train)
	H2 = tf.layers.conv2d(H1,  
					num_outputs=opt.filter_size*2,  
					kernel_size=[opt.sent_len2, 1],  
					activation_fn=conv_acf , 
					padding='VALID', 
					scope=prefix + 'H2', 
					reuse=is_reuse) # batch 1 1 2*Filtersize
	return H2

def deconv_decoder(x, num_layers=2, num_filters=8, filter_sizes=[2, 3], 
			bn=False, training=False,
			timedistributed=False, scope_name="textcnn", 
			reuse=False, activation=tf.nn.relu,
			gated_conv=False, residual=False):

	deconv_op = my_conv1d_transpose.conv1d_transpose

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
				if bn:
					h = tf.contrib.layers.batch_norm(h, is_training=training, scope='bn')
				if j < num_layers - 1:
					h = tf.nn.relu(h)
				res_input = h
		h = tf.nn.relu(h + x)
		conv_blocks.append(h)

	if len(conv_blocks) > 1:
		z = tf.concat(conv_blocks, axis=-1)
	else:
		z = conv_blocks[0]

	# [batch_size, seq, num_filters*len(filter_sizes)]
	return z