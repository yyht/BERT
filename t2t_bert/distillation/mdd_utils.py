import tensorflow as tf
import numpy as np
from distillation.flip_gradient import flip_gradient

EPS = 1e-20

def adv_source_classifier(input_tensor, model_reuse, **kargs):
	
	with tf.variable_scope(kargs.get("scope", "adv_classifier"), reuse=model_reuse):
		input_tensor = tf.layers.dense(input_tensor, 
										input_tensor.get_shape()[-1], 
										tf.nn.tanh,
										name="shared_encoder")
		input_tensor = flip_gradient(input_tensor) # take as bottle-neck features
		logits = tf.layers.dense(input_tensor, kargs.get("num_classes", 5))
	return logits

def margin_disparity_discrepancy(src_f_logit, src_tensor,
								tgt_f_logit, tgt_tensor,
								model_reuse,
								**kargs):
	# source is the student
	# target is the teacher
	gamma = kargs.get("gamma", 4)

	src_f1_logit = adv_source_classifier(src_tensor, model_reuse, **kargs)
	tgt_f1_logit = adv_source_classifier(tgt_tensor, True, **kargs)
	
	pred_label_src_f = tf.argmax(src_f_logit, axis=-1, output_type=tf.int32)
	adv_src_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=tf.nn.log_softmax(src_f1_logit), 
												labels=pred_label_src_f))

	pred_label_tgt_f = tf.argmax(tgt_f_logit, axis=-1, output_type=tf.int32)
	tgt_adv_logit = tf.log(1-tf.exp(tf.nn.log_softmax(tgt_f1_logit))+EPS)

	adv_tgt_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=tgt_adv_logit, 
												labels=pred_label_tgt_f))

	batch_idxs = tf.range(0, tf.shape(src_f_logit)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)
	src_idxs = tf.concat([batch_idxs, tf.expand_dims(pred_label_src_f, axis=-1)], 1)
	logits_src_f = tf.gather_nd(src_f1_logit, src_idxs)
	prob_src_f = tf.exp(tf.nn.log_softmax(logits_src_f))

	tgt_idxs = tf.concat([batch_idxs, tf.expand_dims(pred_label_tgt_f, axis=-1)], 1)
	logits_tgt_f = tf.gather_nd(tgt_f1_logit, tgt_idxs)
	prob_tgt_f = tf.exp(tf.nn.log_softmax(logits_tgt_f))

	return [gamma*adv_src_loss+adv_tgt_loss, prob_src_f, prob_tgt_f]






	
