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

	true_data_loss = (tf.nn.sigmoid_cross_entropy_with_logits(
							logits=true_logits,
							labels=tf.ones_like(true_logits)))
	fake_data_loss = (tf.nn.sigmoid_cross_entropy_with_logits(
							logits=fake_logits,
							labels=tf.zeros_like(fake_logits)))

	if not kargs.get('use_tpu', False):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('true_data_loss', 
							tf.reduce_mean(true_data_loss))
		tf.summary.scalar('fake_data_loss', 
							tf.reduce_mean(fake_data_loss))

	valid_mask = kargs.get('valid_mask', None)
	if_provided = 1
	if valid_mask is None:
		tf.logging.info("====ones valid mask ====")
		shape = bert_utils.get_shape_list(true_data_loss)
		valid_mask = tf.ones(shape=[shape[0]])
		if_provided = 0
	valid_mask = tf.cast(valid_mask, tf.float32)
	if if_provided == 1:
		tf.logging.info("====provided valid mask ====")

	loss = true_data_loss + fake_data_loss
	loss = tf.reduce_sum(loss*valid_mask) / (tf.reduce_sum(valid_mask)+1e-10)

	return loss, true_data_loss, fake_data_loss

def get_residual_ebm_loss(true_ebm_logits, 
					fake_ebm_logits, **kargs):
	
	true_logits = true_ebm_logits
	fake_logits = fake_ebm_logits

	true_data_loss = (tf.nn.sigmoid_cross_entropy_with_logits(
							logits=true_logits,
							labels=tf.ones_like(true_logits)))
	fake_data_loss = (tf.nn.sigmoid_cross_entropy_with_logits(
							logits=fake_logits,
							labels=tf.zeros_like(fake_logits)))

	if not kargs.get('use_tpu', False):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('true_data_loss', 
							tf.reduce_mean(true_data_loss))
		tf.summary.scalar('fake_data_loss', 
							tf.reduce_mean(fake_data_loss))

	valid_mask = kargs.get('valid_mask', None)
	if_provided = 1
	if valid_mask is None:
		tf.logging.info("====ones valid mask ====")
		shape = bert_utils.get_shape_list(true_data_loss)
		valid_mask = tf.ones(shape=[shape[0]])
		if_provided = 0
	valid_mask = tf.cast(valid_mask, tf.float32)
	if if_provided == 1:
		tf.logging.info("====provided valid mask ====")

	loss = true_data_loss + fake_data_loss
	loss = tf.reduce_sum(loss*valid_mask) / (tf.reduce_sum(valid_mask)+1e-10)

	return loss

def get_ebm_mlm_adv_loss(true_ebm_logits, fake_ebm_logits, **kargs):

	d_out_real = true_ebm_logits
	d_out_fake = fake_ebm_logits

	input_shape_list = bert_utils.get_shape_list(d_out_real, 
													expected_rank=[1,2,3])

	gan_type = kargs.get('gan_type', 'standard')

	if gan_type == 'standard':  # the non-satuating GAN loss
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.ones_like(d_out_real)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.ones_like(d_out_fake)
		))
		tf.logging.info("====the non-satuating GAN loss ====")

	elif gan_type == 'JS':  # the vanilla GAN loss
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.ones_like(d_out_real)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = -d_loss_fake
		tf.logging.info("====the vanilla GAN loss ====")

	elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_real, labels=tf.ones_like(d_out_real)
		))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
		))
		d_loss = d_loss_real + d_loss_fake

		g_loss = tf.reduce_mean(-d_out_fake)
		tf.logging.info("====the GAN loss implicitly minimizing KL-divergence ====")

	elif gan_type == 'hinge':  # the hinge loss
		d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
		d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
		d_loss = d_loss_real + d_loss_fake

		g_loss = -tf.reduce_mean(d_out_fake)
		tf.logging.info("====the hinge loss ====")

	elif gan_type == 'tv':  # the total variation distance
		d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
		g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
		tf.logging.info("====the total variation distance ====")

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
		tf.logging.info("====LS-GAN ====")

	elif gan_type == 'RSGAN':  # relativistic standard GAN
		d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
		))
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
		))
		tf.logging.info("====relativistic standard GAN ====")

	else:
		raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator global loss ====")
		tf.summary.scalar('disc_loss', 
							d_loss)
		tf.summary.scalar('gen_loss', 
							g_loss)
		tf.summary.scalar('d_loss_real', 
							d_loss_real)
		tf.summary.scalar('d_loss_fake', 
							d_loss_fake)

	return d_loss, g_loss

def get_noise_loss(true_ebm_logits, true_noise_logits, 
					fake_ebm_logits, fake_noise_logits, 
					num_train_steps, num_warmup_steps,
					**kargs):

	global_step = tf.train.get_or_create_global_step()

	global_steps_int = tf.cast(global_step, tf.int32)
	warmup_steps_int = tf.constant(int(num_warmup_steps), dtype=tf.int32)

	noise_loss_ratio = tf.train.polynomial_decay(
											1.0,
											global_step,
											num_train_steps,
											end_learning_rate=0.0,
											power=1.0,
											cycle=False)
	# when ebm is far from true distribution, we only use kl-divergence to make training stable
	# when ebm is close to true distribution, we apply jsd-divergence for adversarial training
	is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
	noise_ratio = (1.0 - is_warmup) * noise_loss_ratio + is_warmup * 1.0

	loss_mask = kargs.get('loss_mask', None)
	prob_ln = kargs.get('prob_ln', False)
	if loss_mask is not None and not prob_ln:
		tf.logging.info("====apply length normalization for noise kl-divergence ====")
		kl_true_noise_logits = true_noise_logits / (tf.reduce_sum(tf.cast(loss_mask, tf.float32), axis=-1)+1e-10)
	else:
		kl_true_noise_logits = true_noise_logits

	kl_noise_loss = -tf.reduce_mean(kl_true_noise_logits)

	# true_logits = true_ebm_logits - true_noise_logits
	# true_data_loss = (tf.nn.sigmoid_cross_entropy_with_logits(
	# 						logits=true_logits,
	# 						labels=tf.ones_like(true_logits)))
	# first_term = tf.log(2.0) - true_data_loss

	# fake_logits = fake_ebm_logits - fake_noise_logits
	# fake_data_loss = tf.nn.sigmoid_cross_entropy_with_logits(
	# 						logits=fake_logits,
	# 						labels=tf.zeros_like(fake_logits))
	# second_term = tf.log(2.0) - fake_data_loss

	# jsd_noise_loss = tf.reduce_mean(first_term+second_term)
	# if not kargs.get('use_tpu', True):
	# 	tf.summary.scalar('kl_loss', 
	# 					kl_noise_loss)
	# 	tf.summary.scalar('jsd_loss', 
	# 					jsd_noise_loss)
	# 	tf.summary.scalar('noise_ratio', 
	# 					noise_ratio)
	# 	tf.summary.scalar('jsd_loss_first_term', 
	# 					tf.reduce_mean(first_term))
	# 	tf.summary.scalar('jsd_loss_second_term', 
	# 					tf.reduce_mean(second_term))

	# noise_loss = noise_ratio * kl_noise_loss + (1-noise_ratio) * jsd_noise_loss
	noise_loss = kl_noise_loss
	# noise_loss = jsd_noise_loss
	return noise_loss

def ebm_train_metric(true_ebm_logits, fake_ebm_logits, **kargs):

	true_logits = true_ebm_logits
	fake_logits = fake_ebm_logits

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
	# from low to high
	fake_ebm_logll = tf.reduce_mean(fake_ebm_logits)

	return {
		"true_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)),
		"fake_accuracy":tf.reduce_mean(tf.cast(fake_accuracy, tf.float32)),
		"all_accuracy":all_accuracy,
		"true_ebm_logll":true_ebm_logll,
		"fake_ebm_logll":fake_ebm_logll
	}
	
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

def ebm_eval_metric(true_ebm_logits, 
					fake_ebm_logits,
					**kargs):

	true_ebm_logll = tf.metrics.mean(
					values=true_ebm_logits)

	fake_ebm_logll = tf.metrics.mean(
					values=fake_ebm_logits)

	true_logits = true_ebm_logits
	fake_logits = fake_ebm_logits

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

	return {
		"true_ebm_logll":true_ebm_logll,
		"fake_ebm_logll":fake_ebm_logll,
		"true_data_acc":true_data_acc,
		"fake_data_acc":fake_data_acc,
		"all_accuracy":all_accuracy
	}