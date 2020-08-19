import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

def normal_generation(features, 
							adv_features, 
							noise_var,
							target,
							project_norm_type="l2",
							adv_method="freelb",
							**kargs):

	input_mask = tf.cast(tf.not_equal(features['input_ids'], 
								kargs.get('[PAD]', 0)), tf.int32)

	unk_mask = tf.cast(tf.math.equal(features['input_ids'], 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(features['input_ids'], 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(features['input_ids'], 102), tf.float32) # not replace sep
	mask_mask = tf.cast(tf.math.equal(features['input_ids'], 103), tf.float32) # not replace sep
	none_replace_mask =  unk_mask + cls_mask + sep_mask + mask_mask
	noise_mask = tf.cast(input_mask, tf.float32) * (1-none_replace_mask)
	if len(input_shape) == 3:
		tf.logging.info("***** apply seq embedding noise *****")
		noise_mask = tf.expand_dims(noise_mask, axis=-1)
	else:
		tf.logging.info("***** apply embedding table noise *****")

	input_shape = bert_utils.get_shape_list(adv_features)
	noise = tf.random_normal(shape)
	noise *= noise_var
	if len(input_shape) == 3:
		tf.logging.info("***** apply seq embedding noise with noise-mask *****")
		return noise * noise_mask
	else:
		tf.logging.info("***** apply embedding table noise without noise-mask *****")
		return noise

def uniform_generation(features,
						adv_features,
						noise_var,
						project_norm_type="l2",
						adv_method="freelb",
						**kargs):
	input_shape = bert_utils.get_shape_list(adv_features)
	input_mask = tf.cast(tf.not_equal(features['input_ids'], 
								kargs.get('[PAD]', 0)), tf.int32)

	unk_mask = tf.cast(tf.math.equal(features['input_ids'], 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(features['input_ids'], 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(features['input_ids'], 102), tf.float32) # not replace sep
	mask_mask = tf.cast(tf.math.equal(features['input_ids'], 103), tf.float32) # not replace sep
	none_replace_mask =  unk_mask + cls_mask + sep_mask + mask_mask
	noise_mask = tf.cast(input_mask, tf.float32) * (1-none_replace_mask)
	
	if len(input_shape) == 3:
		tf.logging.info("***** apply seq embedding noise *****")
		noise_mask = tf.expand_dims(noise_mask, axis=-1)
	else:
		tf.logging.info("***** apply embedding table noise *****")

	input_lengths = tf.reduce_sum(input_mask, axis=-1)
	if adv_method == "freelb":
		if project_norm_type == "l2":
			noise = tf.random_uniform(shape, -1, 1)
			dims = input_shape[-1]
			if len(input_shape) == 3:
				dims *= input_lengths
			mag = noise_var / tf.sqrt(dims)
			noise *= mag
		elif project_norm_type == "inf":
			noise = tf.random_uniform(shape, -1, 1) 
			noise *= noise_var
	elif adv_method == "rf":
		noise = tf.random_uniform(shape, -1, 1)
		noise *= noise_var
	else:
		noise = tf.random_uniform(shape, -1, 1)
		noise *= noise_var

	if len(input_shape) == 3:
		tf.logging.info("***** apply seq embedding noise with noise-mask *****")
		return noise * noise_mask
	else:
		tf.logging.info("***** apply embedding table noise without noise-mask *****")
		return noise