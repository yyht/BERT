import tensorflow as tf
import numpy as np

import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils
from utils.bert import albert_modules
from metric import tf_metrics

# please refer to
# https://leimao.github.io/article/Noise-Contrastive-Estimation/
def get_ebm_loss(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, **kargs):

	true_logits = true_ebm_logits - true_noise_logits
	fake_logits = fake_ebm_logits - fake_noise_logits

	true_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=true_logits, labels=tf.ones_like(true_logits)
		))

	fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=fake_logits, labels=tf.zeros_like(fake_logits)
		))

	return true_data_loss + fake_data_loss

def get_noise_loss(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, **kargs):
	
	if kargs.get("noise_loss_type", "kl_true_noise") == "kl_noise":
		# minimize the true data distribution between true distribution and noise distribution
		return -tf.reduce_mean(true_noise_logits)

	elif kargs.get("noise_loss_type", "kl_true_noise") == "jsd_noise":
		# followed by FCE: flow based contrastive estimation
		# resembles MLE
		first_term = -(true_ebm_logits + tf.log(1+tf.exp(true_noise_logits-true_ebm_logits)))
		second_term = -tf.log(1+tf.exp(fake_ebm_logits-fake_noise_logits))
		# get mean over batch-dim
		return tf.reduce_mean(first_term+second_term)
	else:
		return -tf.reduce_mean(true_noise_logits)

def ebm_noise_train_metric(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, **kargs):
	
	true_logits = true_ebm_logits - true_noise_logits
	fake_logits = fake_ebm_logits - fake_noise_logits

	true_probs = tf.nn.sigmoid(true_logits)
	fake_probs = tf.nn.sigmoid(fake_logits)

	all_true_probs = tf.oncat([1-true_probs, true_probs], axis=-1)
	all_fake_probs = tf.oncat([1-fake_probs, fake_probs], axis=-1)

	input_shape_list = bert_utils.get_shape_list(all_true_probs, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.ones(batch_size), tf.int32)

	pred_true_label = tf.argmax(all_true_probs, axis=-1)
	pred_fake_label = tf.argmax(all_fake_probs, axis=-1)

	true_accuracy = tf.equal(tf.cast(pred_true_label, tf.int32), tf.cast(true_labels, tf.int32))
	true_accuracy = tf.cast(true_accuracy, tf.float32)
	fake_accuracy = tf.equal(tf.cast(pred_fake_label, tf.int32), tf.cast(fake_labels, tf.int32))
	fake_accuracy = tf.cast(fake_accuracy, tf.float32)

	all_accuracy = tf.reduce_mean(true_accuracy+fake_accuracy)/2

	# from low to high
	true_ebm_logll = tf.reduce_mean(true_ebm_logits)
	true_noise_logll = tf.reduce_mean(true_noise_logits)

	# from low to high
	fake_ebm_logll = tf.reduce_mean(fake_ebm_logits)
	fake_noise_logll = tf.reduce_mean(fake_noise_logits)

	return {
		"true_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)),
		"fake_accuracy":tf.reduce_mean(tf.cast(fake_accuracy, tf.float32)),
		"all_accuracy":all_accuracy,
		"true_ebm_logll":true_ebm_logll,
		"true_noise_logll":true_noise_logll,
		"fake_ebm_logll":fake_ebm_logll,
		"fake_noise_logll":fake_noise_logll
	}

def ebm_noise_eval_metric(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits):
	true_ebm_logll = tf.metrics.mean(
					values=true_ebm_logits)

	true_noise_logll = tf.metrics.mean(
					values=true_noise_logits)

	fake_ebm_logll = tf.metrics.mean(
					values=fake_ebm_logits)

	fake_noise_logll = tf.metrics.mean(
					values=fake_noise_logits)

	true_logits = true_ebm_logits - true_noise_logits
	fake_logits = fake_ebm_logits - fake_noise_logits

	true_probs = tf.nn.sigmoid(true_logits)
	fake_probs = tf.nn.sigmoid(fake_logits)

	all_true_probs = tf.oncat([1-true_probs, true_probs], axis=-1)
	all_fake_probs = tf.oncat([1-fake_probs, fake_probs], axis=-1)

	input_shape_list = bert_utils.get_shape_list(all_true_probs, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.ones(batch_size), tf.int32)

	pred_true_label = tf.argmax(all_true_probs, axis=-1)
	pred_fake_label = tf.argmax(all_fake_probs, axis=-1)

	true_accuracy = tf.equal(tf.cast(pred_true_label, tf.int32), tf.cast(true_labels, tf.int32))
	true_accuracy = tf.cast(true_accuracy, tf.float32)
	fake_accuracy = tf.equal(tf.cast(pred_fake_label, tf.int32), tf.cast(fake_labels, tf.int32))
	fake_accuracy = tf.cast(fake_accuracy, tf.float32)

	true_data_acc = tf.metrics.accuracy(
		labels=true_labels,
		predictions=pred_true_label)

	fake_data_acc = tf.metrics.accuracy(
		labels=fake_labels,
		predictions=pred_fake_label)

	all_data_pred = tf.concat([true_accuracy, fake_accuracy], axis=0)
	all_data_label = tf.concat([true_accuracy, fake_accuracy], axis=0)
	all_accuracy = tf.metrics.accuracy(
		labels=all_data_label,
		predictions=all_data_pred)

	return {
		"true_ebm_logll":true_ebm_logll,
		"true_noise_logll":true_noise_logll,
		"fake_ebm_logll":fake_ebm_logll,
		"fake_noise_logll":fake_noise_logll,
		"true_data_acc":true_data_acc,
		"fake_data_acc":fake_data_acc,
		"all_accuracy":all_accuracy
	}

