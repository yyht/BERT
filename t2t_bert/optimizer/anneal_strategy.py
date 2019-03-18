import tensorflow as tf
import numpy as np

class AnnealStrategy(object):
	def __init__(self, config):
		self.config = config

	def anneal(self, num_train_steps):
		coef_decay = self.config.get("distillation_decay", "polynomial_decay")
		init_coef = tf.constant(value=self.config.get("initial_value", 1.0), shape=[], dtype=tf.float32, name="initial_value")
		end_coef = self.config.get("end_value", 0.0)
		coef = tf.train.polynomial_decay(
										init_coef,
										tf.train.get_or_create_global_step(),
										num_train_steps,
										end_learning_rate=end_coef,
										power=1.0,
										cycle=False)

		return coef
