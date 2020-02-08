import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils
from utils.bert import albert_modules
from metric import tf_metrics


def classifier(config, seq_output,
						input_ids,
						sampled_ids,
						input_mask,
						num_labels,
						dropout_prob,
						**kargs):
	"""
	input_ids: original input ids
	sampled_ids: generated fake ids
	"""
	output_layer = seq_output
	hidden_size = output_layer.shape[-1].value

	unk_mask = tf.cast(tf.equal(input_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.equal(input_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.equal(input_ids, 102), tf.float32) # not replace sep

	none_replace_mask =  unk_mask + cls_mask + sep_mask

	input_mask = tf.cast(input_mask, tf.int32)
	input_mask *= tf.cast(1 - none_replace_mask, tf.int32) # cls, unk, sep are not considered as replace or original

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	if config.get('ln_type', 'postln') == 'preln':
		output_layer = albert_modules.layer_norm(output_layer)
		print('====preln transformer====')
	elif config.get('ln_type', 'postln') == 'postln':
		output_layer = output_layer
		print('====postln transformer====')
	else:
		output_layer = output_layer
		print('====no layer layer_norm====')

	output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

	logits = tf.einsum("abc,dc->abd", output_layer, output_weights)
	logits = tf.nn.bias_add(logits, output_bias) # batch x seq_length x 2

	input_ids = tf.cast(input_ids, tf.int32)

	input_shape_list = bert_utils.get_shape_list(sampled_ids, expected_rank=[2,3])
	if len(input_shape_list) == 3:
		tmp_sampled_ids = tf.argmax(sampled_ids, axis=-1) # batch x seq x vocab
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** gumbel 3-D sampled_ids *******")
	elif len(input_shape_list) == 2:
		tmp_sampled_ids = sampled_ids
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** normal 2-D sampled_ids *******")

	sampled_binary_mask = kargs.get('sampled_binary_mask', None)

	if sampled_binary_mask is not None:
		tf.logging.info("****** loss mask using masked token mask for masked tokens *******")
		loss_mask = sampled_binary_mask
	else:
		tf.logging.info("****** loss mask using input_mask for all tokens *******")
		loss_mask = input_mask

	# ori_sampled_ids = kargs.get('ori_sampled_ids', None)
	# if ori_sampled_ids is not None:
	# 	input_shape_list = bert_utils.get_shape_list(ori_sampled_ids, expected_rank=[2,3])
	# 	if len(input_shape_list) == 3:
	# 		tmp_ori_sampled_ids = tf.argmax(ori_sampled_ids, axis=-1) # batch x seq x vocab
	# 		tmp_ori_sampled_ids = tf.cast(tmp_sampled_ori_ids, tf.int32)
	# 		tf.logging.info("****** gumbel 3-D sampled_ids *******")
	# 	elif len(input_shape_list) == 2:
	# 		tmp_ori_sampled_ids = tf.cast(ori_sampled_ids, tf.int32)
	# 		tf.logging.info("****** normal 2-D sampled_ids *******")

	# 	masked_not_equal_mask = tf.cast(tf.not_equal(input_ids, tmp_ori_sampled_ids), tf.int32)
	# 	masked_not_equal_mask *= tf.cast(input_mask, tf.int32)
	# else:
	# 	masked_not_equal_mask = None

	# if masked_not_equal_mask is not None:
	# 	tf.logging.info("****** loss mask using masked token mask for masked tokens *******")
	# 	loss_mask = masked_not_equal_mask
	# else:
	# 	tf.logging.info("****** loss mask using input_mask for all tokens *******")
	# 	loss_mask = input_mask

	# original:0, replace:1
	not_equal_label_ids = tf.cast(tf.not_equal(input_ids, tmp_sampled_ids), tf.int32)
	not_equal_label_ids *= tf.cast(input_mask, tf.int32)

	if kargs.get('loss', 'cross_entropy') == 'cross_entropy':
		tf.logging.info("====logging discriminator loss using cross entropy ====")
		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													logits=logits,
													labels=tf.stop_gradient(not_equal_label_ids))
	elif kargs.get('loss', 'cross_entropy') == 'focal_loss':
		tf.logging.info("====logging discriminator loss using focal loss ====")
		input_shape_list = bert_utils.get_shape_list(input_ids, expected_rank=2)
		batch_size = input_shape_list[0]
		seq_length = input_shape_list[1]
		not_equal_label_ids_ = tf.reshape(not_equal_label_ids, [batch_size*seq_length])
		logits_ = tf.reshape(logits, [batch_size*seq_length, -1])
		per_example_loss, _ = loss_utils.focal_loss_binary_v2(config, logits_, not_equal_label_ids_)
		per_example_loss = tf.reshape(per_example_loss, [batch_size, seq_length])

	# loss = per_example_loss * tf.cast(loss_mask, tf.float32)
	# loss = tf.reduce_sum(loss) / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))

	equal_label_ids = (1 - tf.cast(not_equal_label_ids, tf.float32)) * tf.cast(loss_mask, tf.float32)
	equal_per_example_loss = per_example_loss * equal_label_ids
	equal_loss = tf.reduce_sum(equal_per_example_loss)
	equal_loss_all = equal_loss / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	equal_loss_output = equal_loss / (1e-10 + tf.reduce_sum(equal_label_ids))

	not_equal_per_example_loss = per_example_loss * tf.cast(not_equal_label_ids, tf.float32)
	not_equal_loss = tf.reduce_sum(not_equal_per_example_loss) # not equal:1, equal:0
	not_equal_loss_all = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	not_equal_loss_output = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(not_equal_label_ids, tf.float32)))

	loss = (equal_loss + not_equal_loss) / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	# loss = equal_loss_output + not_equal_loss_output * 0.1
	tf.logging.info("====discriminator classifier use_tpu %s ====", str(kargs.get('use_tpu', True)))
	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('mask_based_loss', 
							loss)

		loss = per_example_loss * tf.cast(loss_mask, tf.float32)
		loss = tf.reduce_sum(loss) / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))

		tf.summary.scalar('equal_loss', 
							equal_loss/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

		tf.summary.scalar('not_equal_loss', 
							not_equal_loss/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

		tf.summary.scalar('loss_decomposition', 
							loss - (equal_loss+not_equal_loss)/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

	return (loss, logits, per_example_loss)

def global_feature_discriminator(config, input_tensor, labels, reuse=None, **kargs):
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
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		labels = tf.reshape(labels, [-1])
		one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, per_example_loss, log_probs)

def optimal_discriminator(config, true_model_dict, true_features_dict,
						fake_model_dict, fake_features_dict, **kargs):

	alpha = (1-0.15)/0.15

	sampled_ids = fake_features_dict['input_ids']
	input_shape_list = bert_utils.get_shape_list(fake_features_dict["input_ori_ids"], 
													expected_rank=[2,3])
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]

	true_logits = tf.exp(tf.nn.log_softmax(tf.reshape(true_model_dict['masked_lm_log_probs'], [-1, config.vocab_size])))
	fake_logits = tf.exp(tf.nn.log_softmax(tf.reshape(fake_model_dict['masked_lm_log_probs'], [-1, config.vocab_size])))

	labels = tf.reshape(sampled_ids, [-1, 1]) # [batch x seq, 1]
	batch_idxs = tf.range(0, tf.shape(labels)[0])
	batch_idxs = tf.expand_dims(batch_idxs, 1)

	idxs = tf.concat([batch_idxs, labels], 1)
	y_true_pred = tf.gather_nd(true_logits, idxs)
	y_fake_pred = tf.gather_nd(fake_logits, idxs)

	disc_probs = (y_true_pred * (alpha+y_fake_pred)+1e-10) / ((y_fake_pred+alpha*y_true_pred+1e-10))  # batch x seq
	disc_probs = tf.expand_dims(disc_probs, axis=-1) # [batch x seq, 1]
	neg_probs = 1 - disc_probs + 1e-10
	logits = tf.log(tf.concat([disc_probs, neg_probs], axis=-1)+1e-10)

	logits = tf.reshape(logits, [batch_size, seq_length, -1])
	
	input_ids = tf.cast(fake_features_dict['input_ori_ids'], tf.int32)
	unk_mask = tf.cast(tf.equal(input_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.equal(input_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.equal(input_ids, 102), tf.float32) # not replace sep

	none_replace_mask =  unk_mask + cls_mask + sep_mask

	input_mask = fake_features_dict['input_mask']
	input_mask = tf.cast(input_mask, tf.int32)
	input_mask *= tf.cast(1 - none_replace_mask, tf.int32) # cls, unk, sep are not considered as replace or original

	input_shape_list = bert_utils.get_shape_list(sampled_ids, expected_rank=[2,3])
	if len(input_shape_list) == 3:
		tmp_sampled_ids = tf.argmax(sampled_ids, axis=-1) # batch x seq x vocab
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** gumbel 3-D sampled_ids *******")
	elif len(input_shape_list) == 2:
		tmp_sampled_ids = sampled_ids
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)

	sampled_binary_mask = kargs.get('sampled_binary_mask', None)

	if sampled_binary_mask is not None:
		tf.logging.info("****** loss mask using masked token mask for masked tokens *******")
		loss_mask = sampled_binary_mask
	else:
		tf.logging.info("****** loss mask using input_mask for all tokens *******")
		loss_mask = input_mask

	not_equal_label_ids = tf.cast(tf.not_equal(input_ids, tmp_sampled_ids), tf.int32)
	not_equal_label_ids *= tf.cast(loss_mask, tf.int32)

	print(logits.get_shape(), "===disc logits shape==", not_equal_label_ids.get_shape(), "==label ids shape==")

	per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													logits=logits,
													labels=tf.stop_gradient(not_equal_label_ids))

	equal_label_ids = (1 - tf.cast(not_equal_label_ids, tf.float32)) * tf.cast(loss_mask, tf.float32)
	equal_per_example_loss = per_example_loss * equal_label_ids
	equal_loss = tf.reduce_sum(equal_per_example_loss)
	equal_loss_all = equal_loss / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	equal_loss_output = equal_loss / (1e-10 + tf.reduce_sum(equal_label_ids))

	not_equal_per_example_loss = per_example_loss * tf.cast(not_equal_label_ids, tf.float32)
	not_equal_loss = tf.reduce_sum(not_equal_per_example_loss) # not equal:1, equal:0
	not_equal_loss_all = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	not_equal_loss_output = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(not_equal_label_ids, tf.float32)))

	loss = (equal_loss + not_equal_loss) / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	# loss = equal_loss_output + not_equal_loss_output * 0.1
	tf.logging.info("====discriminator classifier use_tpu %s ====", str(kargs.get('use_tpu', True)))
	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('mask_based_loss', 
							loss)

		loss = per_example_loss * tf.cast(loss_mask, tf.float32)
		loss = tf.reduce_sum(loss) / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))

		tf.summary.scalar('equal_loss', 
							equal_loss/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

		tf.summary.scalar('not_equal_loss', 
							not_equal_loss/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

		tf.summary.scalar('loss_decomposition', 
							loss - (equal_loss+not_equal_loss)/(1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32))))

	return (loss, logits, per_example_loss)

def global_gan_loss(config, true_model_dict, true_features_dict,
					fake_model_dict, fake_features_dict, **kargs):

	true_rep = true_model_dict['model'].get_pooled_output()
	fake_rep = fake_model_dict['model'].get_pooled_output()

	input_shape_list = bert_utils.get_shape_list(fake_rep, expected_rank=[2,3])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.zeros(batch_size), tf.int32)
	fake_labels = tf.cast(tf.ones(batch_size), tf.int32)

	(true_loss, true_per_example_loss, true_log_probs) = global_feature_discriminator(config, 
														true_rep, true_labels, 
														reuse=tf.AUTO_REUSE)

	(fake_loss, fake_per_example_loss, fake_log_probs) = global_feature_discriminator(config, 
														fake_rep, fake_labels, 
														reuse=tf.AUTO_REUSE)

	loss = (true_loss + fake_loss) / 2
	per_example_loss = (true_per_example_loss + fake_per_example_loss) / 2

	output_dict = {
		"loss":loss,
		"per_example_loss":per_example_loss,
		"true_loss":true_loss,
		"fake_loss":fake_loss,
		"true_per_example_loss":true_per_example_loss,
		"fake_per_example_loss":fake_per_example_loss,
		"true_log_probs":true_log_probs,
		"fake_log_probs":fake_log_probs
	}

	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator global loss ====")
		tf.summary.scalar('adv_loss', 
							loss)

		tf.summary.scalar('true_loss', 
							true_loss)

		tf.summary.scalar('fake_loss', 
							fake_loss)

	return output_dict

def modified_loss(per_example_loss, logits, input_ids, 
				sampled_ids, input_mask, **kargs):
	input_ids = tf.cast(input_ids, tf.int32)
	input_shape_list = bert_utils.get_shape_list(sampled_ids, expected_rank=[2,3])
	if len(input_shape_list) == 3:
		tmp_sampled_ids = tf.argmax(sampled_ids, axis=-1) # batch x seq x vocab
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** gumbel 3-D sampled_ids *******")
	elif len(input_shape_list) == 2:
		tmp_sampled_ids = sampled_ids
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** normal 2-D sampled_ids *******")

	sampled_binary_mask = kargs.get('sampled_binary_mask', None)

	if sampled_binary_mask is not None:
		tf.logging.info("****** loss mask using masked token mask for masked tokens *******")
		loss_mask = sampled_binary_mask
	else:
		tf.logging.info("****** loss mask using input_mask for all tokens *******")
		loss_mask = input_mask

	not_equal_label_ids = tf.cast(tf.not_equal(input_ids, tmp_sampled_ids), tf.int32)
	not_equal_label_ids *= tf.cast(loss_mask, tf.int32)

	equal_label_ids = (1 - tf.cast(not_equal_label_ids, tf.float32)) * tf.cast(loss_mask, tf.float32)
	equal_per_example_loss = per_example_loss * equal_label_ids
	equal_loss = tf.reduce_sum(equal_per_example_loss)
	equal_loss_all = equal_loss / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))
	equal_loss_self = equal_loss / (1e-10 + tf.reduce_sum(equal_label_ids))

	not_equal_per_example_loss = per_example_loss * tf.cast(not_equal_label_ids, tf.float32)
	not_equal_loss = tf.reduce_sum(not_equal_per_example_loss) # not equal:1, equal:0
	not_equal_loss_all = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(loss_mask, tf.float32)))
	not_equal_loss_self = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(not_equal_label_ids, tf.float32)))

	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('equal_loss_self', 
							equal_loss_self)

		tf.summary.scalar('not_equal_loss_self', 
							not_equal_loss_self)

		tf.summary.scalar('not_equal_num', 
							tf.reduce_sum(not_equal_label_ids))
		tf.summary.scalar('valid_loss_num', 
							tf.reduce_sum(loss_mask))
		tf.summary.scalar('equal_num', 
							tf.reduce_sum(equal_label_ids))

	return [equal_per_example_loss, equal_loss_all, equal_loss_self,
			not_equal_per_example_loss, not_equal_loss_all, not_equal_loss_self]
	
def discriminator_metric_train(per_example_loss, logits, input_ids, sampled_ids,
						input_mask):
	# original:0, replace:1

	input_shape_list = bert_utils.get_shape_list(sampled_ids, expected_rank=[2,3])
	if len(input_shape_list) == 3:
		tmp_sampled_ids = tf.argmax(sampled_ids, axis=-1) # batch x seq x vocab
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** gumbel 3-D sampled_ids *******")
	elif len(input_shape_list) == 2:
		tmp_sampled_ids = sampled_ids
		tmp_sampled_ids = tf.cast(tmp_sampled_ids, tf.int32)
		tf.logging.info("****** normal 2-D sampled_ids *******")

	discriminator_label_ids = tf.not_equal(
						tf.cast(input_ids, tf.int32),
						tf.cast(tmp_sampled_ids, tf.int32)
					)

	equal_label_ids = (1 - tf.cast(discriminator_label_ids, tf.float32)) * tf.cast(input_mask, tf.float32)

	unk_mask = tf.cast(tf.math.equal(input_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(input_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(input_ids, 102), tf.float32) # not replace sep

	none_replace_mask =  unk_mask + cls_mask + sep_mask

	input_mask = tf.cast(input_mask, tf.int32)
	input_mask *= tf.cast(1 - none_replace_mask, tf.int32) # cls, unk, sep are not considered as replace or original

	discriminator_lm_predictions = tf.argmax(
		logits, axis=-1, output_type=tf.int32)

	discriminator_mean_loss = per_example_loss * tf.cast(input_mask, tf.float32)
	discriminator_mean_loss = tf.reduce_sum(discriminator_mean_loss) / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))

	discriminator_lm_accuracy = tf.equal(
						tf.cast(discriminator_lm_predictions, tf.int32),
						tf.cast(discriminator_label_ids, tf.int32)
					)
	discriminator_lm_accuracy = tf.cast(discriminator_lm_accuracy, tf.float32)
	discriminator_lm_accuracy_original = tf.reduce_sum(discriminator_lm_accuracy * tf.cast(equal_label_ids, tf.float32)) / (1e-10 + tf.reduce_sum(tf.cast(equal_label_ids, tf.float32)))
	discriminator_lm_accuracy_diff = tf.reduce_sum(discriminator_lm_accuracy * tf.cast(discriminator_label_ids, tf.float32)) / (1e-10 + tf.reduce_sum(tf.cast(discriminator_label_ids, tf.float32)))
	discriminator_lm_accuracy = tf.reduce_sum(discriminator_lm_accuracy * tf.cast(input_mask, tf.float32)) / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))

	return {
		"discriminator_lm_accuracy": discriminator_lm_accuracy,
		"discriminator_lm_loss": discriminator_mean_loss,
		"discriminator_lm_accuracy_diff":discriminator_lm_accuracy_diff,
		"discriminator_lm_accuracy_original":discriminator_lm_accuracy_original,
		}

def discriminator_metric_global_train(input_dict):
	true_log_probs = input_dict['true_log_probs']
	fake_log_probs = input_dict['fake_log_probs']

	input_shape_list = bert_utils.get_shape_list(true_log_probs, expected_rank=[2])
	batch_size = input_shape_list[0]

	true_labels = tf.cast(tf.zeros(batch_size), tf.int32)
	fake_labels = tf.cast(tf.ones(batch_size), tf.int32)

	pred_true_label = tf.argmax(true_log_probs, axis=-1)
	pred_fake_label = tf.argmax(fake_log_probs, axis=-1)

	true_accuracy = tf.equal(tf.cast(pred_true_label, tf.int32), tf.cast(true_labels, tf.int32))
	fake_accuracy = tf.equal(tf.cast(pred_fake_label, tf.int32), tf.cast(fake_labels, tf.int32))

	return {
		"true_accuracy":tf.reduce_mean(tf.cast(true_accuracy, tf.float32)),
		"fake_accuracy":tf.reduce_mean(tf.cast(fake_accuracy, tf.float32)),

	}
	

def discriminator_metric_eval(per_example_loss, logits, input_ids, sampled_ids,
					input_mask, **kargs):
	# original:0, replace:1
	discriminator_label_ids = tf.not_equal(
		tf.cast(input_ids, tf.int32),
		tf.cast(sampled_ids, tf.int32)
	)
	discriminator_label_ids = tf.cast(discriminator_label_ids, tf.int32)

	unk_mask = tf.cast(tf.math.equal(input_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(input_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(input_ids, 102), tf.float32) # not replace sep

	none_replace_mask =  unk_mask + cls_mask + sep_mask

	input_mask = tf.cast(input_mask, tf.int32)
	input_mask *= tf.cast(1 - none_replace_mask, tf.int32) # cls, unk, sep are not considered as replace or original

	discriminator_lm_predictions = tf.argmax(
		logits, axis=-1, output_type=tf.int32)

	discriminator_label_ids = tf.reshape(discriminator_label_ids, [-1])
	discriminator_lm_predictions = tf.reshape(discriminator_lm_predictions, [-1])

	discriminator_mask = tf.reshape(input_mask, [-1])
	discriminator_accuracy = tf.metrics.accuracy(
		labels=discriminator_label_ids,
		predictions=discriminator_lm_predictions,
		weights=discriminator_mask)

	discriminator_per_example_loss = tf.reshape(per_example_loss, [-1])

	discriminator_mean_loss = tf.metrics.mean(
		values=discriminator_per_example_loss, 
		weights=discriminator_mask)

	output_dict = {
			"discriminator_accuracy":discriminator_accuracy,
			"discriminator_loss":discriminator_mean_loss
	}

	# recall, precision, f1 needs one-hot encoding
	if not kargs.get('use_tpu', True):
		discriminator_f1 = tf_metrics.f1(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask, 
										average="macro")
		discriminator_precison = tf_metrics.precision(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask, 
										average='macro')
		discriminator_recall = tf_metrics.recall(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask, 
										average='macro')
		output_dict['discriminator_f1'] = discriminator_f1
		output_dict['discriminator_precison'] = discriminator_precison
		output_dict['discriminator_recall'] = discriminator_recall
	else:
		discriminator_recall = tf.compat.v1.metrics.recall(
										tf.one_hot(discriminator_label_ids, 2), 
										tf.one_hot(discriminator_lm_predictions, 2),
										weights=discriminator_mask)

		discriminator_precison = tf.compat.v1.metrics.precision(
										tf.one_hot(discriminator_label_ids, 2), 
										tf.one_hot(discriminator_lm_predictions, 2),
										weights=discriminator_mask)
		discriminator_f1 = tf_metrics.f1(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask, 
										average="macro")
		discriminator_f1_original = tf_metrics.f1(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
										pos_indices=[0],
										average="macro")
		discriminator_f1_replaced = tf_metrics.f1(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
										pos_indices=[1],
										average="macro")
		discriminator_precision_original = tf_metrics.precision(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
										pos_indices=[0],
										average="macro")
		discriminator_precision_replaced = tf_metrics.precision(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
										pos_indices=[1],
										average="macro")
		discriminator_recall_original = tf_metrics.recall(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
										pos_indices=[0],
										average="macro")
		discriminator_recall_replaced = tf_metrics.recall(
										discriminator_label_ids,
										discriminator_lm_predictions,
										2, 
										weights=discriminator_mask,
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

	
