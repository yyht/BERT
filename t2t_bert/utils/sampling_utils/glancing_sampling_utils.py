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

def glance_sample(masked_decode_logits,
				masked_lm_positions,
				masked_lm_ids,
				masked_lm_weights,
				input_ids,
				input_ori_ids,
				input_mask,
				num_train_steps,
				vocab_size,
				**kargs):

	# [batch_size, masked_seq_length]
	print(masked_decode_logits.get_shape(), "==masked_decode_logits shape==")
	tf.logging.info(masked_decode_logits)
	tf.logging.info("==masked_decode_logits==info")
	input_shape = bert_utils.get_shape_list(masked_lm_weights)
	# [batch_size, seq_length]
	input_ids_shape = bert_utils.get_shape_list(input_ids)
	global_step = tf.train.get_or_create_global_step()

	# [batch_size x masked_seq_length, vocab_size] => [batch_size, masked_seq_length]
	masked_decode_logits = tf.reshape(masked_decode_logits, input_shape+[vocab_size])
	masked_decoded_labels = tf.argmax(masked_decode_logits, axis=-1, output_type=masked_lm_ids.dtype)
	
	tf.logging.info(masked_decode_logits)
	tf.logging.info("==masked_decode_logits==info")

	tf.logging.info(masked_decoded_labels)
	tf.logging.info("==masked_decoded_labels==info")

	masked_not_equal_ids = tf.not_equal(masked_lm_ids, masked_decoded_labels)
	masked_not_equal_ids = tf.cast(masked_not_equal_ids, tf.float32)
	masked_lm_weights = tf.cast(masked_lm_weights, tf.float32)
	masked_not_equal_ids *= masked_lm_weights

	tf.logging.info(masked_not_equal_ids)
	tf.logging.info("==masked_not_equal_ids==info")

	init_rate = 0.5
	final_rate = 0.5

	if kargs.get("ratio_fn", "linear_decay") == "linear_decay":
		ratio = tf.train.polynomial_decay(
										init_rate,
										global_step,
										num_train_steps,
										end_learning_rate=final_rate,
										power=1.0,
										cycle=False)
		tf.logging.info(" using linear decay from %s to %s"%(str(init_rate), str(final_rate)))
	elif kargs.get("ratio_fn", "linear_decay") == "constant":
		ratio = init_rate
		tf.logging.info("==constant rate== %s"%(str(init_rate)))
	else:
		ratio = init_rate
		tf.logging.info("==constant rate== %s"%(str(init_rate)))

	glance_hamming_distance = tf.reduce_sum(masked_not_equal_ids, axis=-1)
	
	tf.logging.info(glance_hamming_distance)
	tf.logging.info("==glance_hamming_distance==info")

	glance_num = ratio * tf.cast(glance_hamming_distance, dtype=tf.float32)
	glance_ratio = glance_num / (1e-10+tf.reduce_sum(masked_lm_weights, axis=-1))

	tf.logging.info(masked_lm_weights)
	tf.logging.info("==masked_lm_weights==info")

	tf.logging.info(glance_num)
	tf.logging.info("==glance_num==info")

	tf.logging.info(glance_ratio)
	tf.logging.info("==glance_ratio==info")

	sample_probs = tf.ones_like(masked_lm_weights) * masked_lm_weights
	sample_probs = tf.expand_dims(glance_ratio, axis=-1) * tf.cast(sample_probs, tf.float32)

	noise_dist = tf.distributions.Bernoulli(probs=sample_probs, dtype=tf.float32)
	glanced_mask = noise_dist.sample()
	glanced_mask = tf.cast(glanced_mask, tf.float32)

	tf.logging.info(glanced_mask)
	tf.logging.info("==glanced_mask==info")

	none_glanced_lm_weights = (1-glanced_mask) * masked_lm_weights
	glanced_masked_lm_positions = tf.cast(glanced_mask, dtype=masked_lm_positions.dtype) * masked_lm_positions
	none_glanced_masked_lm_positions = (1-tf.cast(glanced_mask, dtype=masked_lm_positions.dtype)) * masked_lm_positions
	none_glanced_masked_lm_ids = (1-tf.cast(glanced_mask, dtype=masked_lm_ids.dtype)) * masked_lm_ids

	tf.logging.info(none_glanced_lm_weights)
	tf.logging.info("==none_glanced_lm_weights==info")

	tf.logging.info(none_glanced_masked_lm_positions)
	tf.logging.info("==none_glanced_masked_lm_positions==info")

	tf.logging.info(none_glanced_masked_lm_ids)
	tf.logging.info("==none_glanced_masked_lm_ids==info")

	glanced_target_mapping = tf.one_hot(glanced_masked_lm_positions, 
										input_ids_shape[-1],
										dtype=tf.int32)
	glanced_target_mask = tf.expand_dims(tf.cast(glanced_masked_lm_positions > 0, dtype=tf.int32) , -1)
	glanced_target_mapping *= glanced_target_mask
	glanced_token_position_mask = tf.reduce_sum(glanced_target_mapping, axis=1)

	tf.logging.info(glanced_token_position_mask)
	tf.logging.info("==glanced_token_position_mask==info")

	output_ids = (1-glanced_token_position_mask) * input_ids + glanced_token_position_mask * input_ori_ids
	
	if not kargs.get('use_tpu', True):
		tf.summary.scalar('glance_ratio', tf.reduce_sum(tf.cast(glance_ratio, tf.float32)) / (1e-10+input_shape[0]))
		tf.summary.scalar('glanced_num', tf.reduce_sum(tf.cast(glanced_mask, tf.float32))  / (1e-10+input_shape[0]))
		tf.summary.scalar('hamming_distance', tf.reduce_sum(tf.cast(glance_hamming_distance, tf.float32))  / (1e-10+input_shape[0]))
		before_glance_not_equal = tf.cast(tf.not_equal(input_ids, input_ori_ids), dtype=tf.float32)
		before_glance_not_equal = tf.reduce_sum(before_glance_not_equal*tf.cast(input_mask, dtype=before_glance_not_equal.dtype))
		tf.summary.scalar('before_glance_not_equal', tf.reduce_sum(before_glance_not_equal)  / (1e-10+input_shape[0]))
	
		after_glance_not_equal = tf.cast(tf.not_equal(output_ids, input_ori_ids), dtype=tf.float32)
		after_glance_not_equal = tf.reduce_sum(after_glance_not_equal*tf.cast(input_mask, dtype=after_glance_not_equal.dtype))
		tf.summary.scalar('after_glance_not_equal', tf.reduce_sum(after_glance_not_equal)  / (1e-10+input_shape[0]))

		tf.summary.scalar('before_glance_lm_weights', tf.reduce_sum(masked_lm_weights)  / (1e-10+input_shape[0]))
		tf.summary.scalar('after_glance_lm_weights', tf.reduce_sum(none_glanced_lm_weights)  / (1e-10+input_shape[0]))

	return [output_ids, 
			none_glanced_masked_lm_positions,
			none_glanced_masked_lm_ids,
			none_glanced_lm_weights]

def sequential_glance_sample(
				sequential_decode_logits,
				input_ids,
				input_ori_ids,
				input_mask,
				num_train_steps,
				vocab_size,
				**kargs):

	# [batch_size, seq_length, vocab_size]
	print(sequential_decode_logits.get_shape(), "==masked_decode_logits shape==")
	tf.logging.info(sequential_decode_logits)
	tf.logging.info("==masked_decode_logits==info")
	# [batch_size, seq_length]
	input_ids_shape = bert_utils.get_shape_list(input_ids)
	global_step = tf.train.get_or_create_global_step()

	# [batch_size , seq_length, vocab_size] => [batch_size, seq_length]
	sequential_decoded_labels = tf.argmax(sequential_decode_logits, axis=-1, output_type=input_ids.dtype)

	tf.logging.info(sequential_decoded_labels)
	tf.logging.info("==sequential_decoded_labels==info")

	sequential_not_equal_ids = tf.not_equal(input_ori_ids, sequential_decoded_labels)
	sequential_not_equal_ids = tf.cast(sequential_not_equal_ids, tf.float32)
	sequential_not_equal_ids *= tf.cast(input_mask, tf.float32)

	tf.logging.info(sequential_not_equal_ids)
	tf.logging.info("==sequential_not_equal_ids==info")

	init_rate = 0.5
	final_rate = 0.5

	if kargs.get("ratio_fn", "linear_decay") == "linear_decay":
		ratio = tf.train.polynomial_decay(
										init_rate,
										global_step,
										num_train_steps,
										end_learning_rate=final_rate,
										power=1.0,
										cycle=False)
		tf.logging.info(" using linear decay from %s to %s"%(str(init_rate), str(final_rate)))
	elif kargs.get("ratio_fn", "linear_decay") == "constant":
		ratio = init_rate
		tf.logging.info("==constant rate== %s"%(str(init_rate)))
	else:
		ratio = init_rate
		tf.logging.info("==constant rate== %s"%(str(init_rate)))

	glance_hamming_distance = tf.reduce_sum(sequential_not_equal_ids, axis=-1)
	
	tf.logging.info(glance_hamming_distance)
	tf.logging.info("==glance_hamming_distance==info")

	glance_num = ratio * tf.cast(glance_hamming_distance, dtype=tf.float32)
	glance_ratio = glance_num / (1e-10+tf.reduce_sum(tf.cast(input_mask, tf.float32), axis=-1))

	tf.logging.info(glance_num)
	tf.logging.info("==glance_num==info")

	tf.logging.info(glance_ratio)
	tf.logging.info("==glance_ratio==info")

	sample_probs = tf.ones_like(tf.cast(input_mask, tf.float32)) * tf.cast(input_mask, tf.float32)
	sample_probs = tf.expand_dims(glance_ratio, axis=-1) * tf.cast(sample_probs, tf.float32)

	noise_dist = tf.distributions.Bernoulli(probs=sample_probs, dtype=tf.float32)
	glanced_mask = noise_dist.sample()

	tf.logging.info(glanced_mask)
	tf.logging.info("==glanced_mask==info")

	glanced_mask = tf.cast(glanced_mask, dtype=input_ori_ids.dtype) * tf.cast(input_mask, dtype=input_ori_ids.dtype)

	output_ids = (1-glanced_mask) * input_ids + glanced_mask * input_ori_ids
	
	if not kargs.get('use_tpu', True):
		tf.summary.scalar('glance_ratio', tf.reduce_sum(tf.cast(glance_ratio, tf.float32)) / (1e-10+input_shape[0]))
		tf.summary.scalar('glanced_num', tf.reduce_sum(tf.cast(glanced_mask, tf.float32))  / (1e-10+input_shape[0]))
		tf.summary.scalar('hamming_distance', tf.reduce_sum(tf.cast(glance_hamming_distance, tf.float32))  / (1e-10+input_shape[0]))
		before_glance_not_equal = tf.cast(tf.not_equal(input_ids, input_ori_ids), dtype=tf.float32)
		before_glance_not_equal = tf.reduce_sum(before_glance_not_equal*tf.cast(input_mask, dtype=before_glance_not_equal.dtype))
		tf.summary.scalar('before_glance_not_equal', tf.reduce_sum(before_glance_not_equal)  / (1e-10+input_shape[0]))
	
		after_glance_not_equal = tf.cast(tf.not_equal(output_ids, input_ori_ids), dtype=tf.float32)
		after_glance_not_equal = tf.reduce_sum(after_glance_not_equal*tf.cast(input_mask, dtype=after_glance_not_equal.dtype))
		tf.summary.scalar('after_glance_not_equal', tf.reduce_sum(after_glance_not_equal)  / (1e-10+input_shape[0]))
	return [output_ids, glanced_mask]


