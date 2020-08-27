import tensorflow as tf
import numpy as np
from utils.bert import bert_utils

def load_confusion_set(vocab_confusion_set,
						vocab_confusion_set_mask):
	vocab_confusion_matrix = []
	with tf.gfile.Open(vocab_confusion_set, "r") as frobj:
		for line in frobj:
			content = line.strip().split()
			vocab_confusion_matrix.append([int(item) for item in content[1:]])

	vocab_confusion_mask = []
	with tf.gfile.Open(vocab_confusion_set_mask, "r") as frobj:
		for line in frobj:
			content = line.strip().split()
			vocab_confusion_mask.append([int(item) for item in content[1:]])
	
	vocab_confusion_matrix = np.array(vocab_confusion_matrix).astype(np.int32)
	vocab_confusion_mask = np.array(vocab_confusion_mask).astype(np.int32)
	tf.logging.info(vocab_confusion_matrix.shape)
	tf.logging.info("==vocab_confusion_matrix==info")

	tf.logging.info(vocab_confusion_mask.shape)
	tf.logging.info("==vocab_confusion_mask==info")
	return vocab_confusion_matrix, vocab_confusion_mask

def confusion_set_sample(rand_ids, tgt_len,
						vocab_confusion_matrix, 
						vocab_confusion_mask,
						switch_ratio=0.5):
	sampled_confusion_set = tf.nn.embedding_lookup(vocab_confusion_matrix, rand_ids)
	sampled_confusion_set_mask = tf.nn.embedding_lookup(vocab_confusion_mask, rand_ids)
	sampled_confusion_set_mask_prob = tf.cast(sampled_confusion_set_mask, tf.float32)
	sampled_confusion_set_mask_prob = sampled_confusion_set_mask_prob / (1e-10+tf.reduce_sum(sampled_confusion_set_mask_prob, axis=-1, keepdims=True))
	
	sampled_ids = tf.multinomial(tf.log(sampled_confusion_set_mask_prob+1e-10), num_samples=1, output_dtype=tf.int32)

	confusion_matrix_shape = bert_utils.get_shape_list(vocab_confusion_matrix, expected_rank=[2])

	sampled_onehot_ids = tf.one_hot(tf.squeeze(sampled_ids, axis=-1), confusion_matrix_shape[-1])
	confusion_output_ids = tf.reduce_sum(tf.cast(sampled_onehot_ids, dtype=rand_ids.dtype)*tf.cast(sampled_confusion_set, dtype=rand_ids.dtype), axis=-1)
	
	# confusion_prob_ids = tf.random.uniform([tgt_len], maxval=2, dtype=rand_ids.dtype)
	confusion_prob = tf.random.uniform([tgt_len], dtype=tf.float32)
	confusion_prob_ids = tf.cast(confusion_prob > switch_ratio, dtype=rand_ids.dtype)
	output_ids = confusion_output_ids * confusion_prob_ids + (1-confusion_prob_ids) * rand_ids

	return output_ids