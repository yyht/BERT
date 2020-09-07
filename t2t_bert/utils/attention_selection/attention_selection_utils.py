import tensorflow as tf
import numpy as np
import math
from utils.bert import bert_utils

"""
https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py
"""

def check_tf_version():
	version = tf.__version__
	print("==tf version==", version)
	if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
		return True
	else:
		return False

def sample_gumbel(shape, samples=1, eps=1e-20): 
	"""Sample from Gumbel(0, 1)"""
	if samples > 1:
		sample_shape = shape + [samples]
	else:
		sample_shape = shape
	U = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
	# return -tf.log(-tf.log(U + eps) + eps)
	return -tf.log(-tf.log(U))

def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)


# def attention_group_sampling(attention_scores, 
# 							attention_mask,
# 							mode,
# 							temperatures=0.1,
# 							sample_type="straight_through"):
# 	"""
# 	# `attention_scores` = [B, N, F, T]
# 	# `attention_mask` = [B, 1, F, T]
# 	"""
# 	valid_scores = tf.expand_dims(attention_scores, axis=-1)
# 	zero_scores = tf.zeros_like(valid_scores, dtype=valid_scores.dtype)

# 	# make sure that 1 is for selected score and 0 is to discard score
# 	gumbel_scores = tf.concat([zero_scores, valid_scores], axis=-1)

# 	if attention_mask is not None:
# 		valid_attention_mask = tf.expand_dims(attention_mask, axis=-1)
# 		zero_mask = tf.zeros_like(valid_attention_mask, dtype=valid_attention_mask.dtype)
# 		gumbel_mask = tf.concat([zero_mask, valid_attention_mask], axis=-1)
	
# 	if mode == tf.estimator.ModeKeys.TRAIN:
# 		valid_scores_shape = bert_utils.get_shape_list(gumbel_scores)

# 		gumbel_noise = sample_gumbel(valid_scores_shape)
# 		gumbel_scores += gumbel_noise

# 		sampled_logprob_temp = tf.exp(tf.nn.log_softmax(gumbel_scores/temperatures, axis=-1))
# 		# [B, N, F, T]
# 		sampled_hard_id = tf.argmax(gumbel_scores, axis=-1)

# 		# [B, N, F, T, 2]
# 		sampled_hard_onehot_id = tf.one_hot(sampled_hard_id, 2)

# 		# [B, N, F, T, 2]
# 		if sample_type == "straight_through":
# 			selected_group = tf.stop_gradient(sampled_hard_onehot_id-sampled_logprob_temp) + (sampled_logprob_temp)
# 		elif sample_type == "soft":
# 			selected_group = sampled_logprob_temp
# 		else:
# 			selected_group = tf.stop_gradient(sampled_hard_onehot_id-sampled_logprob_temp) + (sampled_logprob_temp)

# 		# [B, N, F, T]
# 		selected_attention_scores = tf.reduce_sum(selected_group * gumbel_scores, axis=-1)
# 		if attention_mask is not None:
# 			selected_attention_mask = tf.reduce_sum(tf.cast(selected_group, dtype=gumbel_mask.dtype) * gumbel_mask, axis=-1)

# 	else:
# 		sampled_logprob_temp = tf.exp(tf.nn.log_softmax(gumbel_scores, axis=-1))
# 		# [B, N, F, T]
# 		sampled_hard_id = tf.argmax(sampled_logprob_temp, axis=-1)

# 		# [B, N, F, T, 2]
# 		sampled_hard_onehot_id = tf.one_hot(sampled_hard_id, 2)
# 		# [B, N, F, T]
# 		selected_attention_scores = tf.reduce_sum(sampled_hard_onehot_id * gumbel_scores, axis=-1)
# 		if attention_mask is not None:
# 			selected_attention_mask = tf.reduce_sum(tf.cast(sampled_hard_onehot_id, dtype=gumbel_mask.dtype) * gumbel_mask, axis=-1)

# 	# [B, N, F, T]
# 	if attention_mask is not None:
# 		adder = (1.0 - tf.cast(selected_attention_mask, tf.float32)) * -10000.0
# 	else:
# 		adder = 0.0
# 	# Since we are adding it to the raw scores before the softmax, this is
# 	# effectively the same as removing these entirely.
# 	selected_attention_scores += adder
# 	return selected_attention_scores

def attention_group_sampling(from_tensor, 
							to_tensor,
							attention_mask,
							mode,
							batch_size=None,
							from_seq_length=None,
							to_seq_length=None,
							num_attention_heads=1,
							size_per_head=512,
							initializer_range=0.02,
							key_act=None,
							query_act=None,
							temperatures=0.1,
							sample_type="straight_through",
							**kargs):
	"""
	# `attention_scores` = [B, N, F, T]
	# `attention_mask` = [B, 1, F, T]
	"""

	def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
													 seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])

		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor

	from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
	to_shape = bert_utils.get_shape_list(to_tensor, expected_rank=[2, 3])

	print(from_shape, to_shape, "====from shape, and to shape====")
	tf.logging.info(from_tensor)
	tf.logging.info(to_tensor)
	tf.logging.info("==from tensor, to tensor==")

	if len(from_shape) != len(to_shape):
		raise ValueError(
				"The rank of `from_tensor` must match the rank of `to_tensor`.")

	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
		to_seq_length = to_shape[1]
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None or to_seq_length is None):
			raise ValueError(
					"When passing in rank 2 tensors to attention_layer, the values "
					"for `batch_size`, `from_seq_length`, and `to_seq_length` "
					"must all be specified.")

	from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
	to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

	attention_head_size = size_per_head
	tf.logging.info("==apply attention_original_size==")

	# `query_layer` = [B*F, N*H]
	query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=tf.tanh,
			name="query_switch",
			kernel_initializer=create_initializer(initializer_range))

	# `key_layer` = [B*T, N*H]
	key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * attention_head_size,
			activation=tf.tanh,
			name="key_switch",
			kernel_initializer=create_initializer(initializer_range))

	# `query_layer` = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size,
									 num_attention_heads, 
									 from_seq_length,
									 attention_head_size)

	# `key_layer` = [B, N, T, H]
	key_layer = transpose_for_scores(key_layer, batch_size, 
									num_attention_heads,
									to_seq_length, 
									attention_head_size)
	
	# `attention_scores` = [B, N, F, T]
	attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
	# attention_scores = tf.multiply(attention_scores,
	# 								1.0 / math.sqrt(float(attention_head_size)))
	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_or_create_global_step()

		ratio = tf.train.polynomial_decay(
										10.0,
										global_step,
										100000,
										end_learning_rate=0.1,
										power=1.0,
										cycle=False)

		# ratio = temperatures

		attention_scores_shape = bert_utils.get_shape_list(attention_scores)
		tf.logging.info(attention_scores)
		tf.logging.info("==attention_scores== info")
		gumbel_noise_v1 = sample_gumbel(attention_scores_shape)
		gumbel_noise_v2 = sample_gumbel(attention_scores_shape)
		gumbel_noise = attention_scores+gumbel_noise_v1-gumbel_noise_v2
		sampled_logprob_temp = tf.nn.sigmoid(gumbel_noise/ratio) * tf.cast(attention_mask, dtype=attention_scores.dtype)
		# [B, N, F, T]
		sampled_hard_id = tf.cast(sampled_logprob_temp > 0.5, dtype=attention_scores.dtype)

		# [B, N, F, T]
		if sample_type == "straight_through":
			selected_group = tf.stop_gradient(sampled_hard_id-sampled_logprob_temp) + (sampled_logprob_temp)
			tf.logging.info("==apply straight_through structural_attentions==")
		elif sample_type == "soft":
			selected_group = sampled_logprob_temp
			tf.logging.info("==apply soft structural_attentions==")
		else:
			selected_group = tf.stop_gradient(sampled_hard_id-sampled_logprob_temp) + (sampled_logprob_temp)
			tf.logging.info("==apply straight_through structural_attentions==")
		# [B, N, F, T]
		if attention_mask is not None:
			selected_group = selected_group * tf.cast(attention_mask, dtype=tf.float32)
		else:
			selected_group = selected_group * tf.ones_like(attention_scores)
		adder = (1.0 - attention_mask) * -100000.0 + tf.log(selected_group+1e-20)
	
	else:
		tf.logging.info("==apply hard structural_attentions==")
		sampled_logprob_temp = tf.nn.sigmoid(attention_scores)
		# [B, N, F, T]
		sampled_hard_id = tf.cast(sampled_logprob_temp > 0.5, dtype=attention_scores.dtype)

		# [B, N, F, T]
		if attention_mask is not None:
			selected_group = sampled_hard_id * tf.cast(attention_mask, dtype=tf.float32)
		else:
			selected_group = sampled_hard_id * tf.ones_like(attention_scores)
	
		adder = (1.0 - selected_group) * -100000.0
	# [B, N, F, T]
	# Since we are adding it to the raw scores before the softmax, this is
	# effectively the same as removing these entirely.
	attention_scores += adder
	return attention_scores

