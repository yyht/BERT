
import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

"""
taken from gpt2-ml repo
"""

def get_extra_mask(input_ids, input_ori_ids, 
					exclude_mask, vocab_size,
					**kargs):
	unk_mask = tf.cast(tf.math.equal(input_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(input_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(input_ids, 102), tf.float32) # not replace sep
	input_mask = tf.cast(tf.math.not_equal(input_ids, 0), tf.float32) # not replace sep
	
	valid_mask = input_mask * (1-unk_mask-cls_mask-sep_mask)
	
	input_ori_ids_onehot = tf.one_hot(input_ori_ids, vocab_size) # batch x seq x vocab
	input_ori_ids_onehot = tf.cast(input_ori_ids_onehot, dtype=tf.float32)
	corrupted_mask = tf.cast(tf.not_equal(input_ids, input_ori_ids), tf.float32)
	corrupted_mask = tf.expand_dims(corrupted_mask, axis=-1)
	corrupted_mask *= input_ori_ids_onehot

	# add corrupted true-label
	corrupted_mask = exclude_mask * (1-corrupted_mask)
	if kargs.get("remove_euqal_self", False):
		equal_mask = tf.cast(tf.equal(input_ids, input_ori_ids), tf.float32)
		input_ori_ids =  valid_mask * (equal_mask * 1.0 + (1-equal_mask) * tf.cast(input_ori_ids, tf.float32))
		equal_mask = tf.expand_dims(equal_mask, axis=-1)
		equal_mask = equal_mask * input_ori_ids_onehot
		corrupted_mask = equal_mask + (1-equal_mask) * corrupted_mask
		
		unused_token_onehot = tf.cast(tf.one_hot([1], vocab_size), tf.float32)
		unused_token_onehot = tf.expand_dims(unused_token_onehot, axis=0)
		unused_token_onehot = tf.expand_dims(unused_token_onehot, axis=2)
		
		corrupted_mask *= (1 - unused_token_onehot)

	valid_mask = tf.expand_dims(valid_mask, axis=-1)
	corrupted_mask *= valid_mask

	return corrupted_mask, tf.cast(input_ori_ids, dtype=tf.int32)

def reorder(ref, ref_indices):
	def prepare_fd(fd_indices, sd_dims):
		fd_indices = tf.expand_dims(fd_indices, 1)
		fd_indices = tf.tile(fd_indices, [1, sd_dims])
		return tf.cast(fd_indices, tf.int32)

	fd_indices_range = tf.range(0, limit=tf.shape(ref)[0])
	sd_dims = tf.shape(ref_indices)[1]
	pp = prepare_fd(fd_indices_range, sd_dims)

	indices = tf.stack((prepare_fd(fd_indices_range, sd_dims), ref_indices), axis=2)

	updates_ref = tf.gather_nd(ref, indices)
	return updates_ref

def nucleus_sampling(logits, vocab_size, p=0.9, 
					input_ids=None, input_ori_ids=None,
					**kargs):
	input_shape_list = bert_utils.get_shape_list(logits, expected_rank=[2,3])
	if len(input_shape_list) == 3:
		logits = tf.reshape(logits, (-1, vocab_size))
	probs = tf.nn.softmax(logits, axis=-1)
	# [batch_size, seq, vocab_perm]
	# indices = tf.argsort(probs, direction='DESCENDING')
	indices = tf.contrib.framework.argsort(probs, direction='DESCENDING')

	cumulative_probabilities = tf.math.cumsum(tf.batch_gather(probs, indices), axis=-1, exclusive=False)
	
	# find the top pth index to cut off. careful we don't want to cutoff everything!
	# result will be [batch_size, seq, vocab_perm]
	exclude_mask = tf.logical_not(
	tf.logical_or(cumulative_probabilities < p, tf.range(vocab_size)[None] < 1))
	exclude_mask = tf.cast(exclude_mask, tf.float32)

	indices_v1 = tf.contrib.framework.argsort(indices)
	exclude_mask = reorder(exclude_mask, tf.cast(indices_v1, dtype=tf.int32))
	if len(input_shape_list) == 3:
		exclude_mask = tf.reshape(exclude_mask, input_shape_list)
		logits = tf.reshape(logits, input_shape_list)

	if input_ids is not None and input_ori_ids is not None:
		exclude_mask, input_ori_ids = get_extra_mask(
								input_ids, input_ori_ids, 
								exclude_mask, vocab_size,
								**kargs)

	return logits, exclude_mask, input_ori_ids