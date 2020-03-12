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

	print(true_ebm_logits.get_shape(), true_noise_logits.get_shape(), "=======ebm logits shape====")
	print(fake_ebm_logits.get_shape(), fake_noise_logits.get_shape(), "=======noise logits shape====")

	true_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							logits=true_logits,
							labels=tf.ones_like(true_logits)))
	fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							logits=fake_logits,
							labels=tf.zeros_like(fake_logits)))

	# true_data_loss = -tf.reduce_mean(tf.log(1+tf.exp(true_noise_logits-true_ebm_logits)))
	# fake_data_loss = -tf.reduce_mean(tf.log(1+tf.exp(fake_ebm_logits-fake_noise_logits)))

	if not kargs.get('use_tpu', False):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('true_data_loss', 
							true_data_loss)
		tf.summary.scalar('fake_data_loss', 
							fake_data_loss)

	return (true_data_loss + fake_data_loss)

def get_noise_loss(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, **kargs):
	
	if kargs.get("noise_loss_type", "kl_true_noise") == "kl_noise":
		tf.logging.info("****** optimize noise-dist with kl-divergence between true and noise-dist *******")
		# minimize the true data distribution between true distribution and noise distribution
		return -tf.reduce_mean(true_noise_logits)

	elif kargs.get("noise_loss_type", "kl_true_noise") == "jsd_noise":
		tf.logging.info("****** optimize noise-dist with jsd between true and noise-dist *******")
		# followed by FCE: flow based contrastive estimation
		# resembles MLE
		first_term = -(true_ebm_logits + tf.log(1+tf.exp(true_noise_logits-true_ebm_logits))-tf.log(2.0))
		second_term = -tf.log(1+tf.exp(fake_ebm_logits-fake_noise_logits))+tf.log(2.0)
		# get mean over batch-dim
		return tf.reduce_mean(first_term+second_term)
	else:
		return -tf.reduce_mean(true_noise_logits)

def ebm_noise_train_metric(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, 
					input_ids, sequence_mask, true_noise_seq_logits,
					**kargs):
	
	true_logits = true_ebm_logits - true_noise_logits
	fake_logits = fake_ebm_logits - fake_noise_logits

	print(true_logits.get_shape(), "=====true logits shape==", fake_logits.get_shape())

	true_probs = tf.expand_dims(tf.nn.sigmoid(true_logits), axis=-1)
	fake_probs = tf.expand_dims(tf.nn.sigmoid(fake_logits), axis=-1)

	print("==true_probs shape==", true_probs.get_shape())

	all_true_probs = tf.concat([1-true_probs, true_probs], axis=-1)
	all_fake_probs = tf.concat([1-fake_probs, fake_probs], axis=-1)

	print(all_true_probs.get_shape(), "==all_true_probs shape==")

	input_shape_list = bert_utils.get_shape_list(all_true_probs, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.zeros(batch_size), tf.int32)

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

	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = true_noise_seq_logits[:, :-1] # 1,2,3,<T>, xxx, x

	noise_token_accuracy = tf.equal(
						tf.cast(labels, tf.int32),
						tf.cast(tf.argmax(logits, axis=-1), tf.int32))

	noise_token_accuracy = tf.reduce_sum(tf.cast(noise_token_accuracy, tf.float32) * sequence_mask[:, 1:], axis=-1)
	noise_token_accuracy /= tf.reduce_sum(sequence_mask[:, 1:], axis=-1) # batch

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	per_example_perplexity = tf.reduce_sum(input_id_logits * sequence_mask[:, 1:], axis=-1) # batch
	per_example_perplexity /= tf.reduce_sum(sequence_mask[:, 1:], axis=-1) # batch

	perplexity = tf.reduce_mean(tf.exp(per_example_perplexity))

	if not kargs.get("prob_ln", False):
		tf.logging.info("****** sum of plogprob as sentence probability *******")
		noise_ppl = tf.reduce_mean(tf.exp(-true_noise_logits/tf.reduce_sum(sequence_mask, axis=-1)))
	else:
		tf.logging.info("****** sum of plogprob with length normalization as sentence probability *******")
		noise_ppl = tf.reduce_mean(tf.exp(-true_noise_logits))

	return {
		"true_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)),
		"fake_accuracy":tf.reduce_mean(tf.cast(fake_accuracy, tf.float32)),
		"all_accuracy":all_accuracy,
		"true_ebm_logll":true_ebm_logll,
		"true_noise_logll":true_noise_logll,
		"fake_ebm_logll":fake_ebm_logll,
		"fake_noise_logll":fake_noise_logll,
		"noise_ppl":noise_ppl,
		"noise_token_accuracy":tf.reduce_mean(noise_token_accuracy),
		"noise_ppl_ori":perplexity
	}

def ebm_noise_eval_metric(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits,
					input_ids, sequence_mask, true_noise_seq_logits,
					**kargs):
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

	true_probs = tf.expand_dims(tf.nn.sigmoid(true_logits), axis=-1)
	fake_probs = tf.expand_dims(tf.nn.sigmoid(fake_logits), axis=-1)

	all_true_probs = tf.concat([1-true_probs, true_probs], axis=-1)
	all_fake_probs = tf.concat([1-fake_probs, fake_probs], axis=-1)

	input_shape_list = bert_utils.get_shape_list(all_true_probs, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.ones(batch_size), tf.int32)
	fake_labels = tf.cast(tf.zeros(batch_size), tf.int32)

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

	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = true_noise_seq_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	noise_token_accuracy = tf.metrics.accuracy(
					labels=tf.cast(labels, tf.int32), 
					predictions=tf.cast(tf.argmax(logits, axis=-1), tf.int32),
					weights=sequence_mask[:, 1:])

	perplexity = tf.exp(-true_noise_logits)
	noise_ppl = tf.metrics.mean(values=perplexity)

	return {
		"true_ebm_logll":true_ebm_logll,
		"true_noise_logll":true_noise_logll,
		"fake_ebm_logll":fake_ebm_logll,
		"fake_noise_logll":fake_noise_logll,
		"true_data_acc":true_data_acc,
		"fake_data_acc":fake_data_acc,
		"all_accuracy":all_accuracy,
		"noise_token_accuracy":noise_token_accuracy,
		"noise_ppl":noise_ppl
	}

