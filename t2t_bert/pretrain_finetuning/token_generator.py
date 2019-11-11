import tensorflow as tf
import numpy as np

from utils.bert import bert_utils
from utils.bert import bert_modules, albert_modules

def top_k_logits(logits, k):
	if k == 0:
		# no truncation
		return logits

	def _top_k():
		values, _ = tf.nn.top_k(logits, k=k)
		min_values = values[:, -1, tf.newaxis]
		return tf.where(
			logits < min_values,
			tf.ones_like(logits, dtype=logits.dtype) * -1e10,
			logits,
		)
	return tf.cond(
	   tf.equal(k, 0),
	   lambda: logits,
	   lambda: _top_k(),
	)

def token_generator(config, input_tensor,
					output_weights, 
					input_ids, 
					input_ori_ids,
					input_mask, 
					**kargs):
	
	input_shape_list = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]
	hidden_dims = input_shape_list[2]

	embedding_projection = kargs.get('embedding_projection', None)

	scope = kargs.get('scope', None)
	if scope:
		scope = scope + '/' + 'cls/predictions'
	else:
		scope = 'cls/predictions'

	tf.logging.info("**** mlm generator scope **** %s", str(scope))

	# with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		if config.get('ln_type', 'postln') == 'preln':
			input_tensor = albert_modules.layer_norm(input_tensor)
		elif config.get('ln_type', 'postln') == 'postln':
			input_tensor = input_tensor
		else:
			input_tensor = input_tensor

		if config.get("embedding", "factorized") == "factorized":
			projection_width = config.hidden_size
		else:
			projection_width = config.embedding_size

		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=projection_width,
					activation=albert_modules.get_activation(config.hidden_act),
					kernel_initializer=albert_modules.create_initializer(
							config.initializer_range))

			if config.get('ln_type', 'postln') == 'preln':
				input_tensor = input_tensor
			elif config.get('ln_type', 'postln') == 'postln':
				input_tensor = albert_modules.layer_norm(input_tensor)
			else:
				input_tensor = albert_modules.layer_norm(input_tensor)

		if embedding_projection is not None:
			# batch x seq x hidden, embedding x hidden
			print(input_tensor.get_shape(), embedding_projection.get_shape())
			input_tensor = tf.einsum("abc,dc->abd", input_tensor, embedding_projection)
		else:
			print("==no need for embedding projection==")
			input_tensor = input_tensor

		output_bias = tf.get_variable(
				"output_bias",
				shape=[config.vocab_size],
				initializer=tf.zeros_initializer())
		# batch x seq x embedding
		logits = tf.einsum("abc,dc->abd", input_tensor, output_weights)
		logits = tf.nn.bias_add(logits, output_bias)

		input_shape_list = bert_utils.get_shape_list(logits, expected_rank=3)
		width = input_shape_list[2]

		logits_tempered = logits / config.get("temperature", 1.0)

		flat_logits_tempered = tf.reshape(logits_tempered,
									[batch_size * seq_length, width])

		flat_logits_tempered_topk = top_k_logits(flat_logits_tempered, int(config.vocab_size/2))

		samples = tf.multinomial(flat_logits_tempered_topk, 
								num_samples=config.get('gen_sample', 1), 
								output_dtype=tf.int32)

		label_diff_ids = tf.equal(
						tf.cast(input_ids, tf.int32),
						tf.cast(input_ori_ids, tf.int32)
					)
		label_diff_ids = tf.cast(label_diff_ids, tf.float32)
		print(label_diff_ids, "===label diff ids===")
		tf.summary.scalar('label_diff_ids', 
							tf.reduce_sum(label_diff_ids*tf.cast(input_mask, tf.float32))/tf.reduce_sum(tf.cast(input_mask, tf.float32)))

		if config.get('gen_sample', 1) == 1:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length])
			if kargs.get('mask_method', 'all') == 'only_mask':
				label_diff_ids = tf.cast(label_diff_ids, tf.float32)
				samples = (1 - label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + label_diff_ids * tf.cast(input_ori_ids, tf.float32)
				sampled_input_id = tf.cast(sampled_input_id, tf.int32)
		else:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length, config.get('gen_sample', 1)])
			if kargs.get('mask_method', 'all') == 'only_mask':
				# batch x seq_length x 1
				label_diff_ids = tf.expand_dims(label_diff_ids, axis=-1)
				label_diff_ids = tf.einsum('abc,cd->abd', label_diff_ids, tf.ones((1, model_config.get('gen_sample', 1))))
				# batch x seq_length x 1
				input_ori_ids = tf.expand_dims(input_ori_ids, axis=-1)
				input_ori_ids = tf.einsum('abc,cd->abd', input_ori_ids, tf.ones((1, model_config.get('gen_sample', 1))))
				input_ori_ids = tf.cast(input_ori_ids, tf.float32)

				sampled_input_id = (1 - label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + input_ori_ids * label_diff_ids
				sampled_input_id = tf.cast(sampled_input_id, tf.int32)

		return sampled_input_id

def generator_metric_fn_train(masked_lm_example_loss, masked_lm_log_probs, 
					masked_lm_ids,
					masked_lm_weights, 
					next_sentence_example_loss,
					next_sentence_log_probs, 
					next_sentence_labels,
					**kargs):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
	masked_lm_weights = tf.cast(masked_lm_weights, tf.float32)

	masked_lm_mask = kargs.get('masked_lm_mask', None)
	if masked_lm_mask is not None:
		masked_lm_weights *= tf.cast(masked_lm_mask, tf.float32)

	masked_lm_accuracy = tf.equal(
						tf.cast(masked_lm_ids, tf.int32),
						tf.cast(masked_lm_predictions, tf.int32)
					)
	masked_lm_accuracy = tf.cast(masked_lm_accuracy, tf.int32)*tf.cast(masked_lm_weights, dtype=tf.int32)
	masked_lm_accuracy = tf.reduce_sum(tf.cast(masked_lm_accuracy, tf.float32)) / tf.reduce_sum(masked_lm_weights)
	masked_lm_mean_loss = tf.reduce_sum(masked_lm_example_loss*masked_lm_weights) / tf.reduce_sum(masked_lm_weights)

	if next_sentence_log_probs is not None:
		next_sentence_log_probs = tf.reshape(
				next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
		next_sentence_predictions = tf.argmax(
				next_sentence_log_probs, axis=-1, output_type=tf.int32)
		next_sentence_labels = tf.reshape(next_sentence_labels, [-1])

		next_sentence_accuracy = tf.equal(
							tf.cast(next_sentence_labels, tf.int32),
							tf.cast(next_sentence_predictions, tf.int32)
						)
		next_sentence_accuracy = tf.reduce_mean(tf.cast(next_sentence_accuracy, tf.float32))
		next_sentence_loss = tf.reduce_mean(next_sentence_example_loss)

		return {
			"masked_lm_accuracy": masked_lm_accuracy,
			"masked_lm_loss": masked_lm_mean_loss,
			"next_sentence_accuracy": next_sentence_accuracy,
			"next_sentence_loss": next_sentence_loss,
			"valid_position":tf.reduce_sum(masked_lm_weights)
			}
	else:
		return {
			"masked_lm_accuracy": masked_lm_accuracy,
			"masked_lm_loss": masked_lm_mean_loss,
			"valid_position":tf.reduce_sum(masked_lm_weights)
			}

def generator_metric_fn_eval(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
					masked_lm_weights, next_sentence_example_loss,
					next_sentence_log_probs, next_sentence_labels):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
	masked_lm_accuracy = tf.metrics.accuracy(
		labels=masked_lm_ids,
		predictions=masked_lm_predictions,
		weights=masked_lm_weights)
	masked_lm_mean_loss = tf.metrics.mean(
		values=masked_lm_example_loss, weights=masked_lm_weights)

	if next_sentence_log_probs is not None:

		next_sentence_log_probs = tf.reshape(
			next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
		next_sentence_predictions = tf.argmax(
			next_sentence_log_probs, axis=-1, output_type=tf.int32)
		next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
		next_sentence_accuracy = tf.metrics.accuracy(
			labels=next_sentence_labels, predictions=next_sentence_predictions)
		next_sentence_mean_loss = tf.metrics.mean(
			values=next_sentence_example_loss)

		return {
			"masked_lm_accuracy": masked_lm_accuracy,
			"masked_lm_loss": masked_lm_mean_loss,
			"next_sentence_accuracy": next_sentence_accuracy,
			"next_sentence_loss": next_sentence_mean_loss
			}
	else:
		return {
			"masked_lm_accuracy": masked_lm_accuracy,
			"masked_lm_loss": masked_lm_mean_loss,
			"valid_position":tf.reduce_sum(masked_lm_weights)
			}

	

			