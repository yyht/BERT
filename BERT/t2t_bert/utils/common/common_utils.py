import tensorflow as tf
import numpy as np

def word_dropout(inputs, dropout_rate):
	input_shape = tf.shape(inputs)
	batch_size = input_shape[0]
	n_time_steps = input_shape[1]
	mask = tf.random_uniform((batch_size, n_time_steps, 1)) >= dropout_rate

	w_drop = tf.cast(mask, tf.float32) * inputs
	return w_drop, mask