import tensorflow as tf
import numpy as np
from utils.bert import bert_utils
from distillation.flip_gradient import flip_gradient


def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)

def CPC_Hidden(student_tensor, teacher_tensor, input_mask):

	# input_mask: batch x seq

	teacher_shape = bert_utils.get_shape_list(teacher_tensor[0], expected_rank=[3])
	student_shape = bert_utils.get_shape_list(student_tensor[0], expected_rank=[3])

	with tf.variable_scope("cpc_weights", reuse=tf.AUTO_REUSE): 
		cpc_weights = tf.get_variable(
				"weights", [student_shape[0],teacher_shape[0]],
				initializer=create_initializer(0.02)
				)

	# batch x seq x t_hidden
	student_tensor_proj = tf.einsum("abc,cd->abd", student_tensor[-1], cpc_weights)
	# batch x seq x t_hidden and batch x seq x t_hidden
	# log exp(zt x W x ct)
	# batch x batch x seq
	cpc_tensor = tf.einsum("abd,cbd->acb", student_tensor_proj, teacher_tensor[-1])

	mask = tf.cast(input_mask, tf.float32) # batch x seq

	joint_sample_mask = tf.eye(student_shape[0], dtype=bool)
	joint_sample_mask = tf.expand_dims(joint_sample_mask, axis=-1) # batch x batch x 1

	joint_masked_cpc_tensor = tf.cast(joint_sample_mask, tf.float32) * cpc_tensor
	marginal_masked_cpc_tensor = cpc_tensor

	# got each seq joint term
	joint_term = tf.reduce_sum(joint_masked_cpc_tensor, axis=[1]) # batch x seq

	marginal_term = tf.reduce_logsumexp(marginal_masked_cpc_tensor, axis=[1]) # batch x seq

	log_n = tf.math.log(tf.cast(cpc_tensor.shape[1], logu.dtype))

	marginal_term -= log_n

	return -tf.reduce_sum((joint_term - marginal_term)*mask) / (1e-10 + tf.reduce_sum(mask))

def WPC_Hidden(student_tensor, teacher_tensor, input_mask, opt):
	teacher_shape = bert_utils.get_shape_list(teacher_tensor[0], expected_rank=[3])
	student_shape = bert_utils.get_shape_list(student_tensor[0], expected_rank=[3])

	with tf.variable_scope("wpc_weights", reuse=tf.AUTO_REUSE): 
		cpc_weights = tf.get_variable(
				"weights", [student_shape[0],teacher_shape[0]],
				initializer=create_initializer(0.02)
				)

	flipped_student_tensor = flip_gradient(student_tensor[-1])
	flipped_teacher_tensor = flip_gradient(teacher_tensor[-1])

	# batch x seq x t_hidden
	student_tensor_proj = tf.einsum("abc,cd->abd", flipped_student_tensor, cpc_weights)
	# batch x seq x t_hidden and batch x seq x t_hidden
	# log exp(zt x W x ct)
	# batch x batch x seq
	cpc_tensor = tf.einsum("abd,cbd->acb", student_tensor_proj, flipped_teacher_tensor)

	mask = tf.cast(input_mask, tf.float32) # batch x seq

	joint_sample_mask = tf.eye(student_shape[0], dtype=bool)
	joint_sample_mask = tf.expand_dims(joint_sample_mask, axis=-1) # batch x batch x 1

	joint_masked_cpc_tensor = tf.cast(joint_sample_mask, tf.float32) * cpc_tensor
	marginal_masked_cpc_tensor = cpc_tensor

	# got each seq joint term
	joint_term = tf.reduce_sum(joint_masked_cpc_tensor, axis=[1]) # batch x seq

	marginal_term = tf.reduce_logsumexp(marginal_masked_cpc_tensor, axis=[1]) # batch x seq

	loss = -tf.reduce_sum((joint_term - marginal_term)*mask) / (1e-10 + tf.reduce_sum(mask))

	wpc_grad = opt.compute_gradients(loss, [])
		
	wpc_grad = tf.sqrt(tf.reduce_sum(tf.square(wpc_grad), axis=1))
	wpc_grad_penality = tf.reduce_mean(tf.square(wpc_grad - 1.0) * 0.1)

	return loss + wpc_grad_penality








