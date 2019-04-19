import tensorflow as tf
import numpy as np
from distillation.flip_gradient import flip_gradient

def correlation(x, y):
	x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
	y = y - tf.reduce_mean(y, axis=-1, keepdims=True)
	x = tf.nn.l2_normalize(x, -1)
	y = tf.nn.l2_normalize(y, -1)
	return -tf.reduce_sum(x*y, axis=-1) # higher the better

def kd(x, y):
	x_prob = tf.nn.softmax(x)
	print(x_prob.get_shape(), y.get_shape(), tf.reduce_sum(x_prob * y, axis=-1).get_shape())
	return -tf.reduce_sum(x_prob * y, axis=-1) # higher the better

def mse(x, y):
	x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
	y = y - tf.reduce_mean(y, axis=-1, keepdims=True)
	return tf.reduce_sum((x-y)**2, axis=-1) # lower the better

def logits_distillation(student_tensor, teacher_tensor, kd_type):
	if kd_type == "person":
		return correlation(student_tensor, teacher_tensor)
	elif kd_type == "kd":
		return kd(teacher_tensor, student_tensor)
	elif kd_type == "mse":
		return mse(student_tensor, teacher_tensor)

def feature_distillation(input_tensor, l, 
					domain_label, num_labels,
					dropout_prob):
	feat = flip_gradient(input_tensor, l)
	hidden_size = input_tensor.shape[-1].value

	output_weights = tf.get_variable(
		"output_weights", [num_labels, hidden_size],
		initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
		"output_bias", [num_labels], initializer=tf.zeros_initializer())

	# output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

	logits = tf.matmul(feat, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	logits = tf.nn.log_softmax(logits)
		
	domain_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
											logits=logits, 
											labels=domain_label)
	domain_loss = tf.reduce_mean(domain_example_loss)

	return (domain_loss, domain_example_loss, logits) 