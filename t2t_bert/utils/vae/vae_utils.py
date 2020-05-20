import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

"""
https://kexue.fm/archives/7381
https://arxiv.org/pdf/2004.12585.pdf
1. kl-divergence has a lower bound
2. set lower bound to far from 0 with some in-equalities
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
						initializer=tf.zero_initializer()
						)
		mean_scale = tau + (1 - tau) * tf.nn.sigmoid(scale)
		return normalize_tensor * tf.sqrt(mean_scale+1e-20)

def std_normalize_scale(tensor, 
						is_training, 
						scope, 
						tau=0.5,
						reuse=False,
						**kargs):
	with tf.variable_scope(scope+"/std_scale", reuse=reuse):
		normalize_tensor = tf.layers.batch_normalization(tensor,
									center=False, 
									scale=False,
									training=is_training,
									epsilon=1e-10)
		scale = tf.get_variable("scale", 
						shape=[tensor_shape[-1]],
						dtype=tf.float32,
						initializer=tf.zero_initializer()
						)
		std_scale = (1 - tau) * tf.nn.sigmoid(-scale)
		return normalize_tensor * tf.sqrt(std_scale+1e-20)


def hidden_sampling(z_mean, z_std, **kargs):
	"""重参数采样
	"""
	noise = K.random_normal(shape=tf.shape(z_mean))
	return z_mean + z_std * noise

def reconstruction_loss(decoder_outputs, input_ids, **kargs):

	sequence_mask = tf.to_float(tf.not_equal(input_ids[:, 1:], 
											kargs.get('[PAD]', 0)))

	seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=input_ids[:, :-1], 
						logits=decoder_outputs[:, 1:])
		
	per_example_loss = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
	loss = tf.reduce_mean(per_example_loss)
	return loss

def kl_divergence(z_mean, z_std, **kargs):

	logvars = tf.log(tf.pow(z_std, 2)+1e-20)

	kl_cost = -0.5 * (logvars - tf.square(z_mean) -
        			tf.pow(z_std, 2) + 1.0)
	kl_cost = tf.reduce_sum(kl_cost, axis=-1)
	return tf.reduce_mean(kl_cost)
