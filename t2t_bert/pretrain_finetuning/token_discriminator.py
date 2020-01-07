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

	ori_sampled_ids = kargs.get('ori_sampled_ids', None)
	if ori_sampled_ids is not None:
		input_shape_list = bert_utils.get_shape_list(ori_sampled_ids, expected_rank=[2,3])
		if len(input_shape_list) == 3:
			tmp_ori_sampled_ids = tf.argmax(ori_sampled_ids, axis=-1) # batch x seq x vocab
			tmp_ori_sampled_ids = tf.cast(tmp_sampled_ori_ids, tf.int32)
			tf.logging.info("****** gumbel 3-D sampled_ids *******")
		elif len(input_shape_list) == 2:
			tmp_ori_sampled_ids = tf.cast(ori_sampled_ids, tf.int32)
			tf.logging.info("****** normal 2-D sampled_ids *******")

		masked_not_equal_mask = tf.cast(tf.not_equal(input_ids, tmp_ori_sampled_ids), tf.int32)
		masked_not_equal_mask *= tf.cast(input_mask, tf.int32)
	else:
		masked_not_equal_mask = None
	if masked_not_equal_mask is not None:
		tf.logging.info("****** loss mask using masked token mask for masked tokens *******")
		loss_mask = masked_not_equal_mask
	else:
		tf.logging.info("****** loss mask using input_mask for all tokens *******")
		loss_mask = input_mask

	# original:0, replace:1
	not_equal_label_ids = tf.cast(tf.not_equal(input_ids, tmp_sampled_ids), tf.int32)
	not_equal_label_ids *= tf.cast(input_mask, tf.int32)

	if kargs.get('loss', 'cross_entropy') == 'cross_entropy':
		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													logits=logits,
													labels=tf.stop_gradient(not_equal_label_ids))
	elif kargs.get('loss', 'cross_entropy') == 'focal_loss':
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

	tf.logging.info("====discriminator classifier use_tpu %s ====", str(kargs.get('use_tpu', True)))
	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('mask_based_loss', 
							loss)

		tf.summary.scalar('equal_loss', 
							equal_loss/(1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32))))

		tf.summary.scalar('not_equal_loss', 
							not_equal_loss/(1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32))))

		tf.summary.scalar('loss_decomposition', 
							loss - (equal_loss+not_equal_loss)/(1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32))))

	return (loss, logits, per_example_loss)

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

	not_equal_label_ids = tf.cast(tf.not_equal(input_ids, tmp_sampled_ids), tf.int32)
	not_equal_label_ids *= tf.cast(input_mask, tf.int32)

	equal_label_ids = (1 - tf.cast(not_equal_label_ids, tf.float32)) * tf.cast(input_mask, tf.float32)
	equal_per_example_loss = per_example_loss * equal_label_ids
	equal_loss = tf.reduce_sum(equal_per_example_loss)
	equal_loss_all = equal_loss / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))
	equal_loss_self = equal_loss / (1e-10 + tf.reduce_sum(equal_label_ids))

	not_equal_per_example_loss = per_example_loss * tf.cast(not_equal_label_ids, tf.float32)
	not_equal_loss = tf.reduce_sum(not_equal_per_example_loss) # not equal:1, equal:0
	not_equal_loss_all = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))
	not_equal_loss_self = not_equal_loss / (1e-10 + tf.reduce_sum(tf.cast(not_equal_label_ids, tf.float32)))

	if not kargs.get('use_tpu', True):
		tf.logging.info("====logging discriminator loss ====")
		tf.summary.scalar('equal_loss_self', 
							equal_loss_self)

		tf.summary.scalar('not_equal_loss_self', 
							not_equal_loss_self)

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

def discriminator_metric_eval(per_example_loss, logits, input_ids, sampled_ids,
					input_mask):
	# original:0, replace:1
	discriminator_label_ids = tf.not_equal(
		tf.cast(input_ids, tf.int32),
		tf.cast(sampled_ids, tf.int32)
	)

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

	discriminator_recall = tf.compat.v1.metrics.recall(discriminator_label_ids, 
						discriminator_lm_predictions,
						weights=discriminator_mask)

	discriminator_precision = tf.compat.v1.metrics.precision(discriminator_label_ids, 
						discriminator_lm_predictions,
						weights=discriminator_mask)

	# discriminator_f1 = 2*(discriminator_recall * discriminator_precision) / ( discriminator_recall + discriminator_precision)


	return {
		"discriminator_accuracy":discriminator_accuracy,
		"discriminator_loss":discriminator_mean_loss,
		"discriminator_recall":discriminator_recall,
		"discriminator_precision":discriminator_precision,
	}

	
