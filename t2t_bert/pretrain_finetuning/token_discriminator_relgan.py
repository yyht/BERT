import tensorflow as tf
import numpy as np

import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils
from utils.bert import albert_modules
from metric import tf_metrics

def gradient_penalty(x_real_onehot, x_fake_onehot_appr, config):
	"""compute the gradiet penalty for the WGAN-GP loss"""
	alpha = tf.random_uniform(shape=[config['batch_size'], 1, 1], minval=0., maxval=1.)
	interpolated = alpha * x_real_onehot + (1. - alpha) * x_fake_onehot_appr

	logit = discriminator(x_onehot=interpolated)

	grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
	grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)  # l2 norm

	GP = config['reg_param'] * tf.reduce_mean(tf.square(grad_norm - 1.))

	return GP

def global_discriminator_logits(config, input_tensor, reuse=None, **kargs):
	"""Get loss and log probs for the next sentence prediction."""
	# Simple binary classification. Note that 0 is "next sentence" and 1 is
	# "random sentence". This weight matrix is not used after pre-training.

	scope = kargs.get('scope', None)
	if scope:
		scope = scope + '/' + 'cls/seq_global'
	else:
		scope = 'cls/seq_global'
	tf.logging.info("**** nsp scope **** %s", str(scope))

	# with tf.variable_scope("cls/seq_relationship", reuse=reuse):
	with tf.variable_scope(scope, reuse=reuse):
		output_weights = tf.get_variable(
				"output_weights",
				shape=[2, config.hidden_size],
				initializer=albert_modules.create_initializer(config.initializer_range))
		output_bias = tf.get_variable(
				"output_bias", shape=[2], initializer=tf.zeros_initializer())

		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		
		return logits

def get_losses(d_out_real, d_out_fake, **kargs):
	# 1:original, 0:fake
	
	input_shape_list = bert_utils.get_shape_list(d_out_real, 
													expected_rank=[1,2,3])

	batch_size = input_shape_list[0]
	gan_type = kargs.get('gan_type', 'standard')

	tf.logging.info("**** gan type **** %s", str(gan_type))

	if gan_type == 'standard':  # the non-satuating GAN loss
		d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.cast(tf.zeros(batch_size), tf.int32)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		tf.logging.info("**** gan type **** %s", str(gan_type))
	elif gan_type == 'JS':  # the vanilla GAN loss
		d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.cast(tf.zeros(batch_size), tf.int32)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = -d_loss_fake
		tf.logging.info("**** gan type **** %s", str(gan_type))

	elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
		d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.cast(tf.zeros(batch_size), tf.int32)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = tf.reduce_mean(-d_out_fake)
		tf.logging.info("**** gan type **** %s", str(gan_type))

	elif gan_type == 'hinge':  # the hinge loss
		d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
		d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
		d_loss = d_loss_real + d_loss_fake

		g_loss = -tf.reduce_mean(d_out_fake)
		tf.logging.info("**** gan type **** %s", str(gan_type))

	elif gan_type == 'tv':  # the total variation distance
		d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
		g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
		tf.logging.info("**** gan type **** %s", str(gan_type))

	# elif gan_type == 'wgan-gp':  # WGAN-GP
	# 	d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
	# 	GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
	# 	d_loss += GP

	# 	g_loss = -tf.reduce_mean(d_out_fake)

	elif gan_type == 'LS':  # LS-GAN
		d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
		d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
		d_loss = d_loss_real + d_loss_fake

		g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))
		tf.logging.info("**** gan type **** %s", str(gan_type))

	elif gan_type == 'RSGAN':  # relativistic standard GAN
		d_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_real - d_out_fake, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=d_out_fake - d_out_real, labels=tf.cast(tf.ones(batch_size), tf.int32)
		))
		tf.logging.info("**** gan type **** %s", str(gan_type))

	else:
		raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator global loss ====")
		tf.summary.scalar('disc_loss', 
							d_loss)

		tf.summary.scalar('gen_loss', 
							g_loss)

	return {"gen_loss":g_loss, "disc_loss":d_loss}

def discriminator_metric_train(input_dict):
	# original:0, replace:1

	d_out_real = input_dict['true_logits']
	d_out_fake = input_dict['fake_logits']

	input_shape_list = bert_utils.get_shape_list(d_out_real, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.zeros(batch_size), tf.int32)

	pred_true_label = tf.argmax(d_out_real, axis=-1)
	pred_fake_label = tf.argmax(d_out_fake, axis=-1)

	true_accuracy = tf.equal(tf.cast(pred_true_label, tf.int32), tf.cast(true_labels, tf.int32))
	fake_accuracy = tf.equal(tf.cast(pred_fake_label, tf.int32), tf.cast(fake_labels, tf.int32))

	return {
		"true_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)),
		"fake_accuracy":tf.reduce_mean(tf.cast(fake_accuracy, tf.float32)),
		"all_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)+tf.cast(fake_accuracy, tf.float32))/2

	}

def discriminator_metric_eval(input_dict):

	d_out_real = input_dict['true_logits']
	d_out_fake = input_dict['fake_logits']

	input_shape_list = bert_utils.get_shape_list(d_out_real, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.zeros(batch_size), tf.int32)

	pred_true_label = tf.argmax(d_out_real, axis=-1)
	pred_fake_label = tf.argmax(d_out_fake, axis=-1)

	all_pred_label = tf.concat([pred_true_label, pred_fake_label], axis=0)
	all_true_label = tf.concat([true_labels, fake_labels], axis=0)

	if not kargs.get('use_tpu', True):
		discriminator_f1 = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										average="macro")
		discriminator_precison = tf_metrics.precision(
										all_true_label,
										all_pred_label,
										2, 
										average="macro")
		discriminator_recall = tf_metrics.recall(
										all_true_label,
										all_pred_label,
										2, 
										average="macro")
		discriminator_f1_original = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_f1_replaced = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")
		discriminator_precision_original = tf_metrics.precision(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_precision_replaced = tf_metrics.precision(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")
		discriminator_recall_original = tf_metrics.recall(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_recall_replaced = tf_metrics.recall(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")
		output_dict['discriminator_f1'] = discriminator_f1
		output_dict['discriminator_precison'] = discriminator_precison
		output_dict['discriminator_recall'] = discriminator_recall
		output_dict['discriminator_f1_original'] = discriminator_f1_original
		output_dict['discriminator_f1_replaced'] = discriminator_f1_replaced
		output_dict['discriminator_precision_original'] = discriminator_precision_original
		output_dict['discriminator_precision_replaced'] = discriminator_precision_replaced
		output_dict['discriminator_recall_original'] = discriminator_recall_original
		output_dict['discriminator_recall_replaced'] = discriminator_recall_replaced
	else:
		discriminator_recall = tf.compat.v1.metrics.recall(
										tf.one_hot(all_true_label, 2), 
										tf.one_hot(all_pred_label, 2))

		discriminator_precison = tf.compat.v1.metrics.precision(
										tf.one_hot(all_true_label, 2), 
										tf.one_hot(all_pred_label, 2))
		discriminator_f1 = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										average="macro")
		discriminator_f1_original = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_f1_replaced = tf_metrics.f1(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")
		discriminator_precision_original = tf_metrics.precision(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_precision_replaced = tf_metrics.precision(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")
		discriminator_recall_original = tf_metrics.recall(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[0],
										average="macro")
		discriminator_recall_replaced = tf_metrics.recall(
										all_true_label,
										all_pred_label,
										2, 
										pos_indices=[1],
										average="macro")

		output_dict['discriminator_f1_original'] = discriminator_f1_original
		output_dict['discriminator_f1_replaced'] = discriminator_f1_replaced
		output_dict['discriminator_precision_original'] = discriminator_precision_original
		output_dict['discriminator_precision_replaced'] = discriminator_precision_replaced
		output_dict['discriminator_recall_original'] = discriminator_recall_original
		output_dict['discriminator_recall_replaced'] = discriminator_recall_replaced
		output_dict['discriminator_f1'] = discriminator_f1
		output_dict['discriminator_precison'] = discriminator_precison
		output_dict['discriminator_recall'] = discriminator_recall
	return output_dict