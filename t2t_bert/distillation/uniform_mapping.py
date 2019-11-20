
import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

"""
only implement where to transfer
what to transfer need:
1. attention_score and hidden states score ratio
2. add head weight
"""

def create_attention_mask_from_input_mask_v1(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = bert_utils.get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask_boradcast = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.cast(tf.expand_dims(to_mask, -1), tf.float32)
#     tf.ones(
#             shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask_boradcast

    return mask

def kl_divergence(source_logits, target_logits):
	source_prob = tf.exp(tf.nn.log_softmax(source_logits))
	target_logits = tf.nn.log_softmax(target_logits)

	kl_distance = -tf.reduce_sum(source_prob * target_logits, axis=-1)
	return tf.reduce_mean(kl_distance)

def l2_distance(source_prob, target_prob, axis):
	l2_distance = tf.reduce_sum(tf.pow(source_prob-target_prob, 2.0), axis=(axis))
	return l2_distance

def l1_distance(source_prob, target_prob, axis):
	l1_distance = tf.reduce_sum(tf.abs(source_prob-target_prob), axis=axis)
	return l1_distance

def attention_score_matching(teacher_score, student_score, 
								input_mask,
								match_direction=0):

	# Scalar dimensions referenced here:
	#   B = batch size (number of sequences)
	#   F = `from_tensor` sequence length
	#   T = `to_tensor` sequence length
	#   N = `num_attention_heads`
	#   H = `size_per_head`

	# Take the dot product between "query" and "key" to get the raw
	# attention scores.
	# `attention_scores` = [B, N, F, T]

	mask = create_attention_mask_from_input_mask_v1(input_mask, input_mask)
	mask = tf.expand_dims(mask, axis=[1]) # B x 1 X F x T
	
	if match_direction == 0:
		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_score), len(teacher_score)],
					initializer=tf.constant_initializer(np.ones((len(student_score), len(teacher_score)))/len(teacher_score), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=-1, keepdims=True)

	else:
		print("===apply teacher model to student model==")

		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_score), len(teacher_score)],
					initializer=tf.constant_initializer(np.ones((len(student_score), len(teacher_score)))/len(student_score), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=0, keepdims=True)

	loss = tf.constant(0.0)

	for i in range(len(student_score)):
		student_score_ = student_score[i]
		student_score_ = tf.nn.log_softmax(student_score_)
		student_score_ *= tf.cast(mask, tf.float32)
		for j in range(len(teacher_score)):
			teacher_score_ = teacher_score[j]
			teacher_score_ = tf.nn.log_softmax(teacher_score_)
			teacher_score_ *= tf.cast(mask, tf.float32)
			weight = normalized_weights[i,j] # normalized to [0,1]
			tmp_loss = weight*l1_distance(teacher_score_, student_score_, axis=[0,1,2,3])
			tmp_loss /= tf.reduce_sum(tf.cast(mask, tf.float32))
			loss += tmp_loss
	loss /= (len(student_score)*len(teacher_score))
	return loss

def hidden_matching(teacher_hidden, student_hidden, 
					input_mask,
					match_direction=0):

	teacher_shape = bert_utils.get_shape_list(teacher_hidden[0], expected_rank=[3])
	student_shape = bert_utils.get_shape_list(student_hidden[0], expected_rank=[3])

	# input_mask: batch x seq x 1
	mask = tf.expand_dims(input_mask, axis=-1)

	if match_direction == 0:

		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_hidden), len(teacher_hidden)],
					initializer=tf.constant_initializer(np.ones((len(student_hidden), len(teacher_hidden)))/len(teacher_hidden), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=-1, keepdims=True)

	else:
		print("===apply teacher model to student model==")
		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_hidden), len(teacher_hidden)],
					initializer=tf.constant_initializer(np.ones((len(student_hidden), len(teacher_hidden)))/len(student_hidden), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=0, keepdims=True)

	# B X F X H

	def projection_fn(input_tensor):

		with tf.variable_scope("uniformal_mapping/projection", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
				"output_weights", [student_shape[-1], teacher_shape[-1]],
				initializer=tf.truncated_normal_initializer(stddev=0.02)
				)

			input_tensor_projection = tf.einsum("abc,cd->abd", input_tensor, 
												projection_weights)
			return input_tensor_projection

	loss = tf.constant(0.0)
	for i in range(len(student_hidden)):
		student_hidden_ = student_hidden[i]
		student_hidden_ = projection_fn(student_hidden_)
		student_hidden_ = tf.nn.l2_normalize(student_hidden_, axis=-1)
		student_hidden_ *= tf.cast(mask, tf.float32)
		for j in range(len(teacher_hidden)):
			teacher_hidden_ = teacher_hidden[j]
			teacher_hidden_ = tf.nn.l2_normalize(teacher_hidden_, axis=-1)
			teacher_hidden_ *= tf.cast(mask, tf.float32)
			weight = normalized_weights[i,j] # normalized to [0,1]
			tmp_loss = weight*l1_distance(student_hidden_, teacher_hidden_, axis=[0,1,2])
			tmp_loss /= tf.reduce_sum(tf.cast(mask, tf.float32))
			loss += tmp_loss
	loss /= (len(student_hidden)*len(teacher_hidden))
	return loss

def hidden_cls_matching(teacher_hidden, student_hidden, match_direction=0):

	teacher_shape = bert_utils.get_shape_list(teacher_hidden[0], expected_rank=[3])
	student_shape = bert_utils.get_shape_list(student_hidden[0], expected_rank=[3])

	if match_direction == 0:

		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_hidden), len(teacher_hidden)],
					initializer=tf.constant_initializer(np.ones((len(student_hidden), len(teacher_hidden)))/len(teacher_hidden), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=-1, keepdims=True)

	else:
		print("===apply teacher model to student model==")
		with tf.variable_scope("attention_weights", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
					"attention_score_weights", [len(student_hidden), len(teacher_hidden)],
					initializer=tf.constant_initializer(np.ones((len(student_hidden), len(teacher_hidden)))/len(student_hidden), dtype=tf.float32)
					)
			normalized_weights = tf.abs(projection_weights) / tf.reduce_sum(tf.abs(projection_weights), axis=0, keepdims=True)

	# B X F X H

	def projection_fn(input_tensor):

		with tf.variable_scope("uniformal_mapping/projection", reuse=tf.AUTO_REUSE): 
			projection_weights = tf.get_variable(
				"output_weights", [student_shape[-1], teacher_shape[-1]],
				initializer=tf.truncated_normal_initializer(stddev=0.02)
				)

			input_tensor_projection = tf.einsum("ac,cd->ad", input_tensor, 
												projection_weights)
			return input_tensor_projection

	loss = tf.constant(0.0)
	for i in range(len(student_hidden)):
		student_hidden_ = student_hidden[i][:,0:1,:]
		student_hidden_ = tf.squeeze(student_hidden_, axis=1)
		student_hidden_ = projection_fn(student_hidden_)
		student_hidden_ = tf.nn.l2_normalize(student_hidden_, axis=-1)
		for j in range(len(teacher_hidden)):
			teacher_hidden_ = teacher_hidden[j][:,0:1,:]
			teacher_hidden_ = tf.squeeze(teacher_hidden_, axis=1)
			teacher_hidden_ = tf.nn.l2_normalize(teacher_hidden_, axis=-1)
			weight = normalized_weights[i,j] # normalized to [0,1]
			tmp_loss = weight*l1_distance(student_hidden_, teacher_hidden_, axis=-1)
			loss += tf.reduce_mean(tmp_loss, axis=0)
	loss /= (len(student_hidden)*len(teacher_hidden))
	return loss

