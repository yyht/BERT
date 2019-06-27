import tensorflow as tf
import numpy as np

def huber_loss(labels, predictions, delta=1.0):
	residual = tf.abs(predictions - labels)
	condition = tf.less(residual, delta) # < 1 is true
	small_res = 0.5 * tf.square(residual)
	large_res = delta * residual - 0.5 * tf.square(delta)
	loss = tf.cast(condition, tf.float32) * small_res + (1-tf.cast(condition, tf.float32)) * large_res
	return loss
	# return tf.select(condition, small_res, large_res)

def dist_mean(predictions):
	condition = tf.greater(predictions, tf.zeros_like(predictions)) # >0 mask
	condition = tf.cast(condition, tf.float32)
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

	labels = tf.ones_like(teacher_angle)
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

	labels = tf.ones_like(student_tensor_pd)
	predictions = student_tensor_pd - teacher_tensor_pd
	loss = huber_loss(labels, predictions)

	return tf.reduce_mean(loss)

def RKD(source, target, l = [1e2,2e2]):
	'''
	Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.  
	relational knowledge distillation.
	arXiv preprint arXiv:1904.05068, 2019.
	'''
	with tf.variable_scope('Relational_Knowledge_distillation'):
		def Huber_loss(x,y):
			with tf.variable_scope('Huber_loss'):
				return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.), 
											   tf.square(x-y)/2, tf.abs(x-y)-1/2))
			
		def Distance_wise_potential(x):
			with tf.variable_scope('DwP'):
				x_square = tf.reduce_sum(tf.square(x),-1)
				prod = tf.matmul(x,x,transpose_b=True)
				distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1)+tf.expand_dims(x_square,0) -2*prod, 1e-12))
				mu = tf.reduce_sum(distance)/tf.reduce_sum(tf.where(distance > 0., tf.ones_like(distance), tf.zeros_like(distance)))
				return distance/(mu+1e-8)
			
		def Angle_wise_potential(x):
			with tf.variable_scope('AwP'):
				e = tf.expand_dims(x,0)-tf.expand_dims(x,1)
				e_norm = tf.nn.l2_normalize(e,2)
			return tf.matmul(e_norm, e_norm,transpose_b=True)

		source = tf.nn.l2_normalize(source,1)
		target = tf.nn.l2_normalize(target,1)
		distance_loss = Huber_loss(Distance_wise_potential(source),Distance_wise_potential(target))
		angle_loss    = Huber_loss(   Angle_wise_potential(source),   Angle_wise_potential(target))
		
		return distance_loss*l[0]+angle_loss*l[1]

