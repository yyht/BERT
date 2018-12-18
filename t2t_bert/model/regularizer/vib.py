import tensorflow as tf
import numpy as np
from utils.bert import bert_utils
distributions = tf.contrib.distributions

class VIB(object):
	def __init__(self, config, *args, **kargs):
		self.config = config

	def build_regularizer(self, inputs, *args, **kargs):
		z_mean, z_log_var = inputs
		noise = tf.random_normal(tf.shape(z_mean))
		latent_vector = z_mean + tf.exp(0.5 * z_log_var) * noise

		shape_lst = bert_utils.get_shape_list(z_mean)
		latent_dim = shape_lst[1]

		if self.config.get("kl_type", "original") == "original":
			kl_loss = 0.5 * (tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1.0)
			
		elif self.config.get("kl_type", "original") == "tf_kl":
			q_sigma = tf.exp(0.5 * z_log_var)
			q_z = distributions.Normal(loc=z_mean, scale=q_sigma)
			p_z = distributions.Normal(loc=tf.zeros(latent_dim, dtype=tf.float32),
                               scale=tf.ones(latent_dim, dtype=tf.float32))

			kl_loss = distributions.kl_divergence(q_z, p_z)

		kl_loss =  self.config.get("beta", 0.1) * tf.reduce_sum(kl_loss, axis=-1)
		return kl_loss, latent_vector



