import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils
from utils.bert import albert_modules

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

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	if config.get('ln_type', 'postln') == 'preln':
		output_layer = albert_modules.layer_norm(output_layer)
	elif config.get('ln_type', 'postln') == 'postln':
		output_layer = output_layer
	else:
		output_layer = output_layer

	output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

	logits = tf.einsum("abc,dc->abd", seq_output, output_weights)
	logits = tf.nn.bias_add(logits, output_bias) # batch x seq_length x 2

	input_ids = tf.cast(input_ids, tf.int32)
	sampled_ids = tf.cast(sampled_ids, tf.int32)

	discriminator_label_ids = tf.cast(tf.equal(input_ids, sampled_ids), tf.int32)

	per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits,
												labels=tf.stop_gradient(discriminator_label_ids))
	loss = per_example_loss * tf.cast(input_mask, tf.float32)

	loss = tf.reduce_sum(loss) / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))

	return (loss, logits, per_example_loss)

def discriminator_metric_train(per_example_loss, logits, input_ids, sampled_ids,
						input_mask):
	discriminator_label_ids = tf.equal(
						tf.cast(input_ids, tf.int32),
						tf.cast(sampled_ids, tf.int32)
					)
	discriminator_lm_predictions = tf.argmax(
		logits, axis=-1, output_type=tf.int32)

	discriminator_mean_loss = per_example_loss * tf.cast(input_mask, tf.float32)
	discriminator_mean_loss = tf.reduce_sum(discriminator_mean_loss) / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))

	discriminator_lm_accuracy = tf.equal(
						tf.cast(discriminator_lm_predictions, tf.int32),
						tf.cast(discriminator_label_ids, tf.int32)
					)
	discriminator_lm_accuracy = tf.cast(discriminator_lm_accuracy, tf.float32)
	discriminator_lm_accuracy = tf.reduce_sum(discriminator_lm_accuracy * tf.cast(input_mask, tf.float32)) / (1e-10 + tf.reduce_sum(tf.cast(input_mask, tf.float32)))
	return {
		"discriminator_lm_accuracy": discriminator_lm_accuracy,
		"discriminator_lm_loss": discriminator_mean_loss		
		}

def discriminator_metric_eval(per_example_loss, logits, input_ids, sampled_ids,
					input_mask):
	discriminator_label_ids = tf.equal(
		tf.cast(input_ids, tf.int32),
		tf.cast(sampled_ids, tf.int32)
	)
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

	return {
		"discriminator_accuracy":discriminator_accuracy,
		"discriminator_loss":discriminator_mean_loss
	}

	