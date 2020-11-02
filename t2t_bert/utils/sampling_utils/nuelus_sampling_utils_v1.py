"""
https://github.com/google-research/google-research/blob/master/routing_transformer/sparse_transformer.py
"""

import tensorflow as tf
import numpy as np

def check_tf_version():
	version = tf.__version__
	print("==tf version==", version)
	if int(version.split(".")[0]) >= 2 or int(version.split(".")[1]) >= 15:
		return True
	else:
		return False

def nucleus_sampling(logits, nucleus_sampling):
	"""Nucleus sampling."""
	p = nucleus_sampling
	tf.logging.info("Nucleus sampling top_p = {}".format(p))
	
	if check_tf_version():
		sort_indices = tf.argsort(logits, axis=-1, direction="DESCENDING")
	else:
		sort_indices = tf.contrib.framework.argsort(logits, direction='DESCENDING')
	probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)
	cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
	# The top 1 candidate always will not be masked.
	# This way ensures at least 1 indices will be selected.
	sort_mask = tf.cast(tf.greater(cumprobs, p), logits.dtype)
	batch_indices = tf.tile(
		tf.expand_dims(tf.range(logits.shape[0]), axis=-1),
		[1, logits.shape[1]])
	top_p_mask = tf.scatter_nd(
		tf.stack([batch_indices, sort_indices], axis=-1), sort_mask,
		logits.shape)
	logits -= top_p_mask * logits.dtype.max
	return logits