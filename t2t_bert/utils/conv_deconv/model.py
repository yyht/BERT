
import tensorflow as tf

def dropout(input_tensor, dropout_prob):
	if dropout_prob is None or dropout_prob == 0.0:
		return input_tensor

	output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output

def discriminator(x, W, opt, prefix = 'd_', is_prob = False, is_reuse = None):
	W_norm_d = tf.identity(W)   # deep copy
	tf.stop_gradient(W_norm_d)  # the discriminator won't update W
	if is_prob:
		x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])  # batch L emb
	else:
		x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb

	x_emb = tf.expand_dims(x_emb, 3)   # batch L emb 1

	if opt.layer == 4:
		H = conv_model_4layer(x_emb, opt, 
							prefix=prefix, 
							is_reuse=is_reuse)
	elif opt.layer == 3:
		H = conv_model_3layer(x_emb, opt, 
							prefix=prefix, 
							is_reuse=is_reuse)
	else: # layer == 2
		H = conv_model(x_emb, opt, 
						prefix=prefix, 
						is_reuse=is_reuse)

	logits = discriminator_2layer(H, opt, 
								prefix= prefix, 
								is_reuse = is_reuse)
	return logits, tf.squeeze(H)

def conv_encoder(x_emb, is_train, opt, res, is_reuse = None, prefix = ''):
	if hasattr(opt, 'multiplier'):
		multiplier = opt.multiplier
	else:
		multiplier = 2
	if opt.layer == 4:
		H_enc = conv_model_4layer(x_emb, opt, 
								is_train=is_train, 
								is_reuse=is_reuse, 
								prefix=prefix)
	elif opt.layer == 3:
		H_enc = conv_model_3layer(x_emb, opt, 
								is_train=is_train, 
								multiplier=multiplier, 
								is_reuse=is_reuse, 
								prefix=prefix)
	else:
		H_enc = conv_model(x_emb, opt, 
								is_train=is_train, 
								is_reuse=is_reuse, 
								prefix=prefix)
	return H_enc, res

def deconv_decoder(H_dec, x_org, W_norm, is_train, opt, res, 
					prefix = '', is_reuse = None):
	if hasattr(opt, 'multiplier'):
		multiplier = opt.multiplier
	else:
		multiplier = 2
	# H_dec  batch 1 1 n_gan
	if opt.layer == 4:
		x_rec = deconv_model_4layer(H_dec, opt, 
									is_train=is_train, 
									prefix=prefix, 
									is_reuse=is_reuse)  #  batch L emb 1
	elif opt.layer == 3:
		x_rec = deconv_model_3layer(H_dec, opt, 
									is_train=is_train, 
									multiplier=multiplier, 
									prefix=prefix, 
									is_reuse=is_reuse)  #  batch L emb 1
	elif opt.layer == 0:
		x_rec = deconv_model_3layer(H_dec, opt, 
									prefix=prefix, 
									is_reuse=is_reuse)  #  batch L emb 1
	else:
		x_rec = deconv_model(H_dec, opt, 
							is_train=is_train, 
							prefix=prefix, 
							is_reuse=is_reuse)  #  batch L emb 1
	print("Decoder len %d Output len %d" % (x_rec.get_shape()[1], x_org.get_shape()[1]))
	tf.assert_equal(x_rec.get_shape()[1], x_org.get_shape()[1])
	x_rec_norm = normalizing(x_rec, 2)    # batch L emb

	x_temp = tf.reshape(x_org, [-1,])
	if hasattr(opt, 'attentive_emb') and opt.attentive_emb:
		emb_att = tf.get_variable(prefix+'emb_att', [1,opt.embed_size], initializer = tf.constant_initializer(1.0, dtype=tf.float32))
		prob_logits = tf.tensordot(tf.squeeze(x_rec_norm), emb_att*W_norm, [[2],[1]])  # c_blv = sum_e x_ble W_ve
	else:
		prob_logits = tf.tensordot(tf.squeeze(x_rec_norm), W_norm, [[2],[1]])  # c_blv = sum_e x_ble W_ve

	prob = tf.nn.log_softmax(prob_logits*opt.L, dim=-1, name=None)
	
	rec_sent = tf.squeeze(tf.argmax(prob,2))
	prob = tf.reshape(prob, [-1,opt.n_words])

	idx = tf.range(opt.batch_size * opt.sent_len)

	all_idx = tf.transpose(tf.stack(values=[idx,x_temp]))
	all_prob = tf.gather_nd(prob, all_idx)

	gen_temp = tf.cast(tf.reshape(rec_sent, [-1,]), tf.int32)
	gen_idx = tf.transpose(tf.stack(values=[idx,gen_temp]))
	gen_prob = tf.gather_nd(prob, gen_idx)

	res['rec_sents'] = rec_sent

	if opt.discrimination:
		logits_real, _ = discriminator(x_org, W_norm, opt)
		prob_one_hot = tf.nn.log_softmax(prob_logits*opt.L, dim=-1, name=None)
		logits_syn, _ = discriminator(tf.exp(prob_one_hot), W_norm, opt, is_prob = True, is_reuse = True)

		res['prob_r'] =  tf.reduce_mean(tf.nn.sigmoid(logits_real))
		res['prob_f'] = tf.reduce_mean(tf.nn.sigmoid(logits_syn))

		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
					 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_syn), logits = logits_syn))
	else:
		loss = -tf.reduce_mean( all_prob)
	return loss, res

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


def conv_model_3layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True, multiplier = 2):
	#XX = tf.reshape(X, [-1, , 28, 1])
	#X shape: batchsize L emb 1
	if opt.reuse_cnn:
		biasInit = opt.cnn_b
		weightInit = opt.cnn_W
	else:
		biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
		weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

	X = regularization(X, opt,  
					prefix=prefix + 'reg_X', 
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
					scope=prefix + 'H1_3', 
					reuse=is_reuse)  # batch L-3 1 Filtersize

	H1 = regularization(H1, opt, 
						prefix=prefix + 'reg_H1', 
						is_reuse=is_reuse, 
						is_train=is_train)
	H2 = tf.layers.conv2d(H1,  
						num_outputs=opt.filter_size*multiplier,  
						kernel_size=[opt.filter_shape, 1], 
						stride=[opt.stride[1],1],  
						biases_initializer=biasInit, 
						activation_fn=None, 
						padding='VALID', 
						scope=prefix + 'H2_3', 
						reuse=is_reuse)
	H2 = regularization(H2, opt,  
						prefix=prefix + 'reg_H2', 
						is_reuse=is_reuse, 
						is_train=is_train)
	H3 = tf.layers.conv2d(H2,  
						num_outputs= (num_outputs if num_outputs else opt.n_gan),  
						kernel_size=[opt.sent_len3, 1], 
						activation_fn=tf.nn.tanh , 
						padding='VALID', 
						scope=prefix + 'H3_3', 
						reuse=is_reuse) # batch 1 1 2*Filtersize

	return H3

def conv_model_4layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True):
	#XX = tf.reshape(X, [-1, , 28, 1])
	#X shape: batchsize L emb 1
	if opt.reuse_cnn:
		biasInit = opt.cnn_b
		weightInit = opt.cnn_W
	else:
		biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
		weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

	X = regularization(X, opt,  prefix= prefix + 'reg_X', is_reuse=is_reuse, is_train=is_train)
	H1 = tf.layers.conv2d(X,  
						num_outputs = opt.filter_size,  
						kernel_size = [opt.filter_shape, opt.embed_size], 
						stride = [opt.stride[0],1],  
						weights_initializer = weightInit, 
						biases_initializer = biasInit, 
						activation_fn = None, 
						padding = 'VALID', 
						scope = prefix + 'H1_3', 
						reuse = is_reuse)  # batch L-3 1 Filtersize

	H1 = regularization(H1, opt, prefix= prefix + 'reg_H1', is_reuse= is_reuse, is_train = is_train)
	H2 = tf.layers.conv2d(H1,  num_outputs=opt.filter_size*2,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H2_3', reuse = is_reuse)

	H2 = regularization(H2, opt, prefix= prefix + 'reg_H2', is_reuse= is_reuse, is_train = is_train)
	H3 = tf.layers.conv2d(H2,  num_outputs=opt.filter_size*4,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[2],1],  biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H3_3', reuse = is_reuse)
	H3 = regularization(H3, opt, prefix= prefix + 'reg_H3', is_reuse= is_reuse, is_train = is_train)
	H4 = tf.layers.conv2d(H3,  num_outputs= (num_outputs if num_outputs else opt.n_gan),  kernel_size=[opt.sent_len4, 1], activation_fn=conv_acf , padding = 'VALID', scope = prefix + 'H4', reuse = is_reuse) # batch 1 1 2*Filtersize
	return H4

dec_acf = tf.nn.relu #tf.nn.tanh
dec_bias = None # tf.constant_initializer(0.001, dtype=tf.float32)

def deconv_model(H, opt, prefix = '', is_reuse= None, is_train = True):
	biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)

	H2t = H

	H2t = regularization(H2t, opt, prefix= prefix + 'reg_H_dec', is_reuse= is_reuse, is_train = is_train)
	H1t = tf.layers.conv2d_transpose(H2t, num_outputs=opt.filter_size,  kernel_size=[opt.sent_len2, 1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope =  prefix + 'H1_t', reuse = is_reuse)

	H1t = regularization(H1t, opt, prefix= prefix + 'reg_H1_dec', is_reuse= is_reuse, is_train = is_train)
	Xhat = tf.layers.conv2d_transpose(H1t, num_outputs=1,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1], biases_initializer=dec_bias, activation_fn=dec_acf, padding = 'VALID',scope = prefix + 'Xhat_t', reuse = is_reuse)
	return Xhat

def deconv_model_3layer(H, opt, prefix = '', is_reuse= None, is_train = True, multiplier = 2):
	#XX = tf.reshape(X, [-1, , 28, 1])
	#X shape: batchsize L emb 1
	biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)

	H3t = H

	H3t = regularization(H3t, opt, prefix= prefix + 'reg_H_dec', is_reuse= is_reuse, is_train = is_train)
	H2t = tf.layers.conv2d_transpose(H3t, num_outputs=opt.filter_size*multiplier,  kernel_size=[opt.sent_len3, 1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H2_t_3', reuse = is_reuse)

	H2t = regularization(H2t, opt, prefix= prefix + 'reg_H2_dec', is_reuse= is_reuse, is_train = is_train)
	H1t = tf.layers.conv2d_transpose(H2t, num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H1_t_3', reuse = is_reuse)

	H1t = regularization(H1t, opt, prefix= prefix + 'reg_H1_dec', is_reuse= is_reuse, is_train = is_train)
	Xhat = tf.layers.conv2d_transpose(H1t, num_outputs=1,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  biases_initializer=dec_bias, activation_fn=dec_acf, padding = 'VALID',scope = prefix + 'Xhat_t_3', reuse = is_reuse)

	return Xhat

def deconv_model_4layer(H, opt, prefix = '', is_reuse= None, is_train = True):
	#XX = tf.reshape(X, [-1, , 28, 1])
	#X shape: batchsize L emb 1
	biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)

	H4t = H

	H4t = regularization(H4t, opt, prefix= prefix + 'reg_H_dec', is_reuse= is_reuse, is_train = is_train)
	H3t = tf.layers.conv2d_transpose(H4t, num_outputs=opt.filter_size*4,  kernel_size=[opt.sent_len4, 1],   biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H3_t_3', reuse = is_reuse)

	H3t = regularization(H3t, opt, prefix= prefix + 'reg_H3_dec', is_reuse= is_reuse, is_train = is_train)
	H2t = tf.layers.conv2d_transpose(H3t, num_outputs=opt.filter_size*2,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[2],1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H2_t_3', reuse = is_reuse)

	H2t = regularization(H2t, opt, prefix= prefix + 'reg_H2_dec', is_reuse= is_reuse, is_train = is_train)
	H1t = tf.layers.conv2d_transpose(H2t, num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H1_t_3', reuse = is_reuse)

	H1t = regularization(H1t, opt, prefix= prefix + 'reg_H1_dec', is_reuse= is_reuse, is_train = is_train)
	Xhat = tf.layers.conv2d_transpose(H1t, num_outputs=1,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  biases_initializer=dec_bias, activation_fn=dec_acf, padding = 'VALID',scope = prefix + 'Xhat_t_3', reuse = is_reuse)
	return Xhat