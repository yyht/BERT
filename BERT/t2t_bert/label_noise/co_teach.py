import tensorflow as tf
import numpy as np
from utils.bert import bert_utils


def co_teach(ori_per_example_loss, coteach_per_example_loss,
			global_step, num_train_step, epoch_k, epoch, ratio_rate):

	global_step_to_epoch = tf.cast(global_step, tf.float32)/tf.cast(num_train_step, tf.float32)*tf.cast(epoch, tf.float32)
	global_step_to_epoch = tf.cast(global_step_to_epoch, tf.int32)
	
	ratio = tf.cast(global_step_to_epoch, tf.float32)/tf.cast(epoch_k, tf.float32) * ratio_rate
	new_ratio_rate = (1 - tf.minimum(ratio, ratio_rate))

	loss_shape = bert_utils.get_shape_list(ori_per_example_loss, 1)

	# topk, k numbers
	topk = tf.cast(new_ratio_rate*loss_shape[0], tf.int32)

	# ori loss topk values and incides
	ori_topk_kv = -tf.nn.top_k(-ori_per_example_loss, topk)
	# coteach loss topk values and incides
	coteach_topk_kv = -tf.nn.top_k(-coteach_per_example_loss, topk)

	# ori loss using coteach topk indices
	ori_loss_from_coteach_topk = tf.gather_nd(params=ori_per_example_loss,
    									indices=tf.expand_dims(coteach_topk_kv.indices, -1)
    									)

	# coteach loss using ori topk indices
	coteach_loss_from_ori_topk = tf.gather_nd(params=coteach_per_example_loss,
    									indices=tf.expand_dims(ori_topk_kv.indices, -1))

	ori_loss = tf.reduce_mean(ori_loss_from_coteach_topk)
	coteach_loss = tf.reduce_mean(coteach_loss_from_ori_topk)

	return ori_loss, coteach_loss