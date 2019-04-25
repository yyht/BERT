import tensorflow as tf
import numpy as np
from distillation.flip_gradient import flip_gradient

EPS = 1e-20

def adv_source_classifier(input_tensor, model_reuse, **kargs):
	input_tensor = flip_gradient(input_tensor)
	with tf.variable_scope(kargs.get("scope", "adv_classifier"), reuse=model_reuse):
		input_tensor = tf.layers.dense(input_tensor, 
										tf.shape(input_tensor)[-1], 
										tf.nn.tanh,
										name="shared_encoder")
		logits = tf.layers.dense(input_tensor, kargs.get("num_classes", 5))
		logits = tf.nn.log_softmax(logits, axis=-1)
	return logits

def margin_disparity_discrepancy(src_f_logit, src_tensor,
								tgt_f_logit, tgt_tensor,
								model_reuse,
								**kargs):
	gamma = kargs.get("gamma", 4)
	src_f_logit = tf.nn.log_softmax(src_f_logit, axis=-1)
	tgt_f_logit = tf.nn.log_softmax(tgt_f_logit, axis=-1)

	src_f1_logit = adv_source_classifier(src_tensor, model_reuse, **kargs)
	tgt_f1_logit = adv_source_classifier(tgt_tensor, True, **kargs)
	
	batch_idxs = tf.range(0, tf.shape(src_f_logit)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	pred_label_src_f = tf.argmax(src_f_logit, axis=-1, output_type=tf.int32)
	pred_label_src_f = tf.expand_dims(pred_label_src_f, axis=1)

	src_idxs = tf.concat([batch_idxs, pred_label_src_f], 1)
	logits_src_f = tf.gather_nd(src_f1_logit, src_idxs)

	batch_idxs = tf.range(0, tf.shape(tgt_f_logit)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	pred_label_tgt_f = tf.argmax(tgt_f_logit, axis=-1, output_type=tf.int32)
	pred_label_tgt_f = tf.expand_dims(pred_label_tgt_f, axis=1)

	src_idxs = tf.concat([batch_idxs, pred_label_tgt_f], 1)
	logits_tgt_f = tf.log(1 - tf.exp(tf.gather_nd(tgt_f1_logit, src_idxs))+EPS)

	return [-tf.reduce_mean(gamma*logits_src_f+logits_tgt_f), src_f1_logit, tgt_f1_logit]






	
