import tensorflow as tf
import numpy as np

from utils.bert import bert_utils

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

def attention_group_sampling(attention_scores, 
							attention_mask,
							mode,
							temperatures=0.1,
							sample_type="straight_through"):
	"""
	# `attention_scores` = [B, N, F, T]
	# `attention_mask` = [B, 1, F, T]
	"""
	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_or_create_global_step()

		# ratio = tf.train.polynomial_decay(
		# 								10.0,
		# 								global_step,
		# 								100000,
		# 								end_learning_rate=0.01,
		# 								power=1.0,
		# 								cycle=False)

		ratio = temperatures

		attention_scores_shape = bert_utils.get_shape_list(attention_scores)
		tf.logging.info(attention_scores)
		tf.logging.info("==attention_scores== info")
		gumbel_noise_v1 = sample_gumbel(attention_scores_shape)
		gumbel_noise_v2 = sample_gumbel(attention_scores_shape)
		log_sigmoid_logits = -tf.nn.softplus(-attention_scores)
		gumbel_noise = log_sigmoid_logits+gumbel_noise_v1-gumbel_noise_v2
		sampled_logprob_temp = tf.nn.sigmoid(gumbel_noise/ratio)
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
			selected_attention_mask = selected_group * tf.cast(attention_mask, dtype=tf.float32)
		else:
			selected_attention_mask = selected_group * tf.ones_like(attention_scores)

	else:
		sampled_logprob_temp = tf.nn.sigmoid(attention_scores)
		# [B, N, F, T]
		sampled_hard_id = tf.cast(sampled_logprob_temp > 0.5, dtype=attention_scores.dtype)

		# [B, N, F, T]
		if attention_mask is not None:
			selected_attention_mask = sampled_hard_id * tf.cast(attention_mask, dtype=tf.float32)
		else:
			selected_attention_mask = sampled_hard_id * tf.ones_like(attention_scores)
	# [B, N, F, T]
	adder = (1.0 - tf.cast(selected_attention_mask, tf.float32)) * -1e20
	# Since we are adding it to the raw scores before the softmax, this is
	# effectively the same as removing these entirely.
	attention_scores += adder
	return attention_scores

