import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

"""
https://kexue.fm/archives/7381
https://arxiv.org/pdf/2004.12585.pdf
1. kl-divergence has a lower bound
2. set lower bound to far from 0 with some in-equalities

batch-normalization:
1. for each dimension, (u-mean)/stdvar to normalize each dimension

"""

def mean_normalize_scale(tensor, 
						is_training, 
						scope, 
						tau=0.5,
						reuse=False,
						**kargs):

	tensor_shape = bert_utils.get_shape_list(tensor, expected_rank=[2,3])
	with tf.variable_scope(scope+"/mean_scale", reuse=reuse): 
		normalize_tensor = tf.layers.batch_normalization(tensor,
									center=False, 
									scale=False,
									training=is_training,
									epsilon=1e-10)

		scale = tf.get_variable("scale", 
						shape=[tensor_shape[-1]],
						dtype=tf.float32,
						initializer=tf.zeros_initializer()
						)
		mean_scale = tau + (1 - tau) * tf.nn.sigmoid(scale)
		return normalize_tensor * tf.sqrt(mean_scale+1e-20)

def std_normalize_scale(tensor, 
						is_training, 
						scope, 
						tau=0.5,
						reuse=False,
						**kargs):
	tensor_shape = bert_utils.get_shape_list(tensor, expected_rank=[2,3])
	with tf.variable_scope(scope+"/std_scale", reuse=reuse):
		normalize_tensor = tf.layers.batch_normalization(tensor,
									center=False, 
									scale=False,
									training=is_training,
									epsilon=1e-10)
		scale = tf.get_variable("scale", 
						shape=[tensor_shape[-1]],
						dtype=tf.float32,
						initializer=tf.zeros_initializer()
						)
		std_scale = (1 - tau) * tf.nn.sigmoid(-scale)
		return normalize_tensor * tf.sqrt(std_scale+1e-20)


def hidden_sampling(z_mean, z_std, **kargs):
	"""
	hidden resampling
	"""
	noise = tf.random_normal(shape=tf.shape(z_mean))
	return z_mean + z_std * noise

def sequence_reconstruction_loss(decoder_outputs, input_ids, name="",
						**kargs):

	sequence_mask = tf.to_float(tf.not_equal(input_ids[:, 1:], 
											kargs.get('[PAD]', 0)))

	seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=input_ids[:, 1:], 
						logits=decoder_outputs[:, :-1])
		
	per_example_loss = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
	loss = tf.reduce_mean(per_example_loss)
	use_tpu = 1 if kargs.get('use_tpu', False) else 0
	if not use_tpu:
		tf.summary.scalar(name+"/reconstruction_loss", loss)
		tf.logging.info("****** vae-reconstruction_loss-summary ******")
	return loss

def kl_loss(z_mean, z_std, anneal_steps, name="",
				**kargs):
	"""
	z_mean: [batch_size, hidden_dims]
	z_std : [batch_size, hidden_dims]
	"""

	if kargs.get('kl_anneal', 'kl_anneal') == 'kl_anneal':
		global_step = tf.train.get_or_create_global_step()

		kl_anneal_ratio = tf.train.polynomial_decay(
												1.0,
												global_step,
												anneal_steps,
												end_learning_rate=0.0,
												power=1.0,
												cycle=False)
		kl_anneal_ratio = 1.0 - kl_anneal_ratio
	else:
		kl_anneal_ratio = 1.0

	logvars = tf.log(tf.pow(z_std, 2)+1e-20)

	per_example_kl_loss = -0.5 * (logvars - tf.pow(z_mean, 2) -
								tf.pow(z_std, 2) + 1.0)
	per_example_kl_loss = tf.reduce_sum(per_example_kl_loss, axis=-1)
	kl_loss = tf.reduce_mean(per_example_kl_loss*kl_anneal_ratio)

	use_tpu = 1 if kargs.get('use_tpu', False) else 0		
	if not use_tpu:
		tf.summary.scalar(name+"/kl_loss", kl_loss)
		tf.summary.scalar(name+"/kl_anneal_ratio", kl_anneal_ratio)
		tf.logging.info("****** vae-kl-summary ******")

	return kl_loss

def tokenid2tf(input_ids, vocab_size, **kargs):
	input_mask = tf.cast(tf.not_equal(input_ids, kargs.get('[PAD]', 0)), 
						tf.float32)
	input_mask = tf.expand_dims(input_mask, axis=-1)
	# one_hot_input_ids = tf.one_hot(input_ids, depth=vocab_size)
	if input_ids.shape.ndims == 2:
		input_ids = tf.expand_dims(input_ids, axis=[-1])
	input_shape = bert_utils.get_shape_list(input_ids)
	flat_input_ids = tf.reshape(input_ids, [-1])
	one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
	one_hot_input_ids = tf.reshape(one_hot_input_ids,
				input_shape[0:-1] + [input_shape[-1] * vocab_size])

	# [batch, seq, vocab_size]
	output = tf.cast(one_hot_input_ids, tf.float32) * input_mask
	# [batch, vocab_size]
	term_count = tf.reduce_sum(output, axis=1)
	# [batch, vocab_size]
	term_binary = tf.minimum(tf.reduce_sum(output, 1), 1)
	term_freq = tf.reduce_sum(output, axis=1) / (1e-10+tf.reduce_sum(output, axis=(1, 2), keepdims=True))
	return term_count, term_binary, term_freq

def bow_loss(input_ids, latent_variables, 
			bow_size, vocab_size, is_training,
			name,
			**kargs):

	term_count, term_binary, term_freq = tokenid2tf(input_ids, vocab_size, **kargs)

	bow_h = tf.layers.dense(latent_variables, 
							bow_size, 
							activation=tf.tanh)
	if is_training:
		dropout = 0.1
	else:
		dropout = 0.0
	bow_h = tf.nn.dropout(bow_h, 1 - dropout)
	bow_logits = tf.layers.dense(bow_h, vocab_size, name="bow_logits")

	if kargs.get('bow_loss', "term_binary") == 'term_binary':
		logits = tf.log(tf.nn.sigmoid(bow_logits)+1e-10)
		bow_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=term_binary, logits=bow_logits)
		bow_loss = tf.reduce_mean(tf.reduce_sum(bow_loss, -1))
		tf.logging.info("****** term_binary_bow_loss ******")
	elif kargs.get('bow_loss', "term_binary") == 'term_count':
		logits = tf.nn.log_softmax(bow_logits)
		bow_loss = -tf.matmul(logits, term_count)
		bow_loss = tf.reduce_mean(tf.reduce_sum(bow_loss, -1))
		tf.logging.info("****** term_count_bow_loss ******")
	elif kargs.get('bow_loss', "term_binary") == 'term_freq':
		logits = tf.nn.log_softmax(bow_logits)
		if kargs.get('bow_loss_type', "mse") == 'mse':
			bow_loss = tf.power(logits - term_freq, 2)
		elif kargs.get('bow_loss_type', "mse") == 'l1':
			bow_loss = tf.abs(logits - term_freq)
		else:
			bow_loss = tf.power(logits - term_freq, 2)
		bow_loss = tf.reduce_mean(tf.reduce_sum(bow_loss, -1))
	else:
		logits = tf.log(tf.nn.sigmoid(bow_logits)+1e-10)
		bow_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=term_binary, logits=bow_logits)
		bow_loss = tf.reduce_mean(tf.reduce_sum(bow_loss, -1))
		tf.logging.info("****** term_binary_bow_loss ******")
	use_tpu = 1 if kargs.get('use_tpu', False) else 0		
	if not use_tpu:
		tf.summary.scalar(name+"/bow_loss", bow_loss)
		tf.logging.info("****** vae-kl-summary ******")
	return bow_loss, logits