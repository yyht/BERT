import tensorflow as tf
import numpy as np

def huber_loss(labels, predictions, delta=1.0):
	residual = tf.abs(predictions - labels)
	condition = tf.less(residual, delta)
	small_res = 0.5 * tf.square(residual)
	large_res = delta * residual - 0.5 * tf.square(delta)
	return tf.select(condition, small_res, large_res)

def dist_mean(predictions):
	condition = tf.greater(predictions, tf.zeros_like(predictions)) # >0 mask
	dist = tf.reduce_sum(condition * predictions)/(tf.reduce_sum(condition)+1e-20)
	return dist

def rkd_angle_loss(student_tensor, teacher_tensor):
	# batch_size x dims: student_tensor, teacher_tensor
	# 1 x batch_size x dims
	# batch_size x 1 x dims
	student_tensor_squeeze = tf.expand_dims(student_tensor, axis=0) - tf.expand_dims(student_tensor, axis=1)
	norm_student_tensor = tf.nn.l2_normalize(student_tensor_squeeze, axis=-1) # batch_size x batch_size
	student_angle = tf.matmul(norm_student_tensor, tf.transpose(norm_student_tensor, perm=(1,2,0)))

	teacher_tensor_squeeze = tf.expand_dims(teacher_tensor, axis=0) - tf.expand_dims(teacher_tensor, axis=1)
	norm_teacher_tensor = tf.nn.l2_normalize(teacher_tensor_squeeze, axis=-1) # batch_size x batch_size
	teacher_angle = tf.matmul(norm_teacher_tensor, tf.transpose(norm_teacher_tensor, perm=(1,2,0)))

	labels = tf.onse_like(teacher_angle)
	predictions = student_angle - teacher_angle
	loss = huber_loss(labels, predictions)

	return tf.reduce_mean(loss)

def rkd_distance_loss(student_tensor, teacher_tensor):
	# batch_size x dims: student_tensor, teacher_tensor
	# 1 x batch_size x dims
	# batch_size x 1 x dims
	student_tensor_squeeze = tf.expand_dims(student_tensor, axis=0) - tf.expand_dims(student_tensor, axis=1)
	teacher_tensor_squeeze = tf.expand_dims(teacher_tensor, axis=0) - tf.expand_dims(teacher_tensor, axis=1)

	student_tensor_pd = tf.sqrt(tf.reduce_sum(tf.pow(student_tensor_squeeze, 2), axis=-1)) # batch_size x batch_size
	teacher_tensor_pd = tf.sqrt(tf.reduce_sum(tf.pow(teacher_tensor_squeeze, 2), axis=-1)) # batch_size x batch_size

	student_tensor_pd /= dist_mean(student_tensor_pd)
	teacher_tensor_pd /= dist_mean(teacher_tensor_pd)

	labels = tf.onse_like(student_tensor_pd)
	predictions = student_tensor_pd - teacher_tensor_pd
	loss = huber_loss(labels, predictions)

	return tf.reduce_mean(loss)

