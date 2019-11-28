import tensorflow as tf
import numpy as np

from utils.bert import bert_utils
from utils.bert import bert_modules, albert_modules

def random_input_ids_generation(config,
							input_ori_ids,
							input_mask,
							**kargs):

	mask_id = kargs.get('mask_id', 103)
	valid_vocab = kargs.get('valid_vocab', 105)

	input_ori_ids = tf.cast(input_ori_ids, tf.int32)
	input_mask = tf.cast(input_mask, tf.int32)

	unk_mask = tf.cast(tf.math.equal(input_ori_ids, 100), tf.float32) # not replace unk
	cls_mask =  tf.cast(tf.math.equal(input_ori_ids, 101), tf.float32) # not replace cls
	sep_mask = tf.cast(tf.math.equal(input_ori_ids, 102), tf.float32) # not replace sep

	none_replace_mask =  unk_mask + cls_mask + sep_mask

	input_shape_list = bert_utils.get_shape_list(input_ori_ids, expected_rank=2)
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]

	must_have_one = tf.cast(tf.expand_dims(tf.eye(seq_length)[4], axis=[0]), tf.int32) # batch x seq_length
	must_have_one *= input_mask * (1 - tf.cast(none_replace_mask, tf.int32))
	sample_probs = tf.ones_like(input_ori_ids) * input_mask * (1 - tf.cast(none_replace_mask, tf.int32))
	sample_probs = 0.2 * tf.cast(sample_probs, tf.float32) + 0.8 * tf.cast(must_have_one, tf.float32) # mask 15% token

	noise_dist = tf.distributions.Bernoulli(probs=sample_probs, dtype=tf.float32)
	sampled_binary_mask = noise_dist.sample()
	sampled_binary_mask = tf.cast(sampled_binary_mask, tf.float32)

	# mask_binary_probs = 0.8 * sampled_binary_mask # use 80% [mask] for masked token
	# mask_noise_dist = tf.distributions.Bernoulli(probs=mask_binary_probs, dtype=tf.float32)
	# sampled_mask_binary_mask = mask_noise_dist.sample()
	# sampled_mask_binary_mask = tf.cast(sampled_mask_binary_mask, tf.float32)

	# replace_binary_probs = 0.5 * (sampled_binary_mask - sampled_mask_binary_mask) # use 10% [mask] to replace token
	# replace_noise_dist = tf.distributions.Bernoulli(probs=replace_binary_probs, dtype=tf.float32)
	# sampled_replace_binary_mask = replace_noise_dist.sample()
	# sampled_replace_binary_mask = tf.cast(sampled_replace_binary_mask, tf.float32)

	# ori_binary_probs = 1.0 * (sampled_binary_mask - sampled_mask_binary_mask - sampled_replace_binary_mask)
	# ori_noise_dist = tf.distributions.Bernoulli(probs=ori_binary_probs, dtype=tf.float32)
	# sampled_ori_binary_mask = ori_noise_dist.sample()
	# sampled_ori_binary_mask = tf.cast(sampled_ori_binary_mask, tf.float32)

	replace_binary_probs = 0.10 * (sampled_binary_mask) # use 10% [mask] to replace token
	replace_noise_dist = tf.distributions.Bernoulli(probs=replace_binary_probs, dtype=tf.float32)
	sampled_replace_binary_mask = replace_noise_dist.sample()
	sampled_replace_binary_mask = tf.cast(sampled_replace_binary_mask, tf.float32)

	ori_binary_probs = 0.10 * (sampled_binary_mask - sampled_replace_binary_mask)
	ori_noise_dist = tf.distributions.Bernoulli(probs=ori_binary_probs, dtype=tf.float32)
	sampled_ori_binary_mask = ori_noise_dist.sample()
	sampled_ori_binary_mask = tf.cast(sampled_ori_binary_mask, tf.float32)

	# mask_binary_probs = 0.85 * (sampled_binary_mask - sampled_replace_binary_mask - sampled_ori_binary_mask) # use 80% [mask] for masked token
	# mask_noise_dist = tf.distributions.Bernoulli(probs=mask_binary_probs, dtype=tf.float32)
	# sampled_mask_binary_mask = mask_noise_dist.sample()
	# sampled_mask_binary_mask = tf.cast(sampled_mask_binary_mask, tf.float32)

	sampled_mask_binary_mask = (sampled_binary_mask - sampled_replace_binary_mask - sampled_ori_binary_mask)
	sampled_mask_binary_mask = tf.cast(sampled_mask_binary_mask, tf.float32)
	
	# sampled_replace_binary_mask *=  (1 - tf.cast(none_replace_mask, tf.float32)) 
	# sampled_replace_binary_mask *= tf.cast(input_mask, tf.float32)

	# sampled_mask_binary_mask *=  (1 - tf.cast(none_replace_mask, tf.float32)) 
	# sampled_mask_binary_mask *= tf.cast(input_mask, tf.float32)
	
	# sampled_ori_binary_mask *=  (1 - tf.cast(none_replace_mask, tf.float32)) 
	# sampled_ori_binary_mask *= tf.cast(input_mask, tf.float32)

	vocab_sample_logits = tf.random.uniform(
							[batch_size, seq_length, config.vocab_size],
							minval=0.0,
							maxval=1.0,
							dtype=tf.float32)

	flatten_vocab_sample_logits = tf.reshape(vocab_sample_logits, 
											[batch_size*seq_length, -1])

	sample_vocab_ids = tf.multinomial(flatten_vocab_sample_logits, 
								num_samples=config.get('gen_sample', 1), 
								output_dtype=tf.int32)

	sample_vocab_ids = tf.reshape(sample_vocab_ids, [batch_size, seq_length])
	sample_vocab_ids = tf.cast(sample_vocab_ids, tf.float32)
	input_ori_ids = tf.cast(input_ori_ids, tf.float32)

	output_input_ids = mask_id * tf.cast(sampled_mask_binary_mask, tf.float32) * tf.ones_like(input_ori_ids)
	output_input_ids += sample_vocab_ids * tf.cast(sampled_replace_binary_mask, tf.float32)
	output_input_ids += (1 - tf.cast(sampled_mask_binary_mask + sampled_replace_binary_mask, tf.float32)) * input_ori_ids
	output_sampled_binary_mask = sampled_mask_binary_mask + sampled_replace_binary_mask + sampled_ori_binary_mask

	output_sampled_binary_mask = tf.cast(output_sampled_binary_mask, tf.int32)

	return [tf.cast(output_input_ids, tf.int32), 
				output_sampled_binary_mask]

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

		# flat_logits_tempered_topk = top_k_logits(flat_logits_tempered, int(config.vocab_size/2))

		samples = tf.multinomial(flat_logits_tempered, 
								num_samples=config.get('gen_sample', 1), 
								output_dtype=tf.int32)

		sampled_binary_mask = kargs.get('sampled_binary_mask', None)
		if sampled_binary_mask is not None:
			label_diff_ids =  sampled_binary_mask # 0 for original and 1 for replace
		else:
			label_diff_ids = tf.not_equal(
							tf.cast(input_ids, tf.int32),
							tf.cast(input_ori_ids, tf.int32) # 0 for original and 1 for replace
						)
		label_diff_ids = tf.cast(label_diff_ids, tf.float32)
		print(label_diff_ids, "===label diff ids===")
		if kargs.get('summary_debug', False):
			tf.summary.scalar('label_diff_ids', 
							tf.reduce_sum(label_diff_ids*tf.cast(input_mask, tf.float32))/tf.reduce_sum(tf.cast(input_mask, tf.float32)))
		
		if config.get('gen_sample', 1) == 1:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length])
			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				label_diff_ids = tf.cast(label_diff_ids, tf.float32)
				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - label_diff_ids) * tf.cast(input_ori_ids, tf.float32)
				sampled_input_id = tf.cast(sampled_input_id, tf.int32)
		else:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length, config.get('gen_sample', 1)])
			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				# batch x seq_length x 1
				label_diff_ids = tf.expand_dims(label_diff_ids, axis=-1)
				label_diff_ids = tf.einsum('abc,cd->abd', label_diff_ids, tf.ones((1, model_config.get('gen_sample', 1))))
				# batch x seq_length x 1
				input_ori_ids = tf.expand_dims(input_ori_ids, axis=-1)
				input_ori_ids = tf.einsum('abc,cd->abd', input_ori_ids, tf.ones((1, model_config.get('gen_sample', 1))))
				input_ori_ids = tf.cast(input_ori_ids, tf.float32)

				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - input_ori_ids) * label_diff_ids
				sampled_input_id = tf.cast(sampled_input_id, tf.int32)

				input_mask = tf.expand_dims(input_mask, axis=-1)
				input_mask = tf.einsum('abc,cd->abd', input_mask, tf.ones((1, model_config.get('gen_sample', 1))))
				input_mask = tf.cast(input_mask, tf.float32)

		sampled_not_equal_id = tf.not_equal(
				tf.cast(sampled_input_id, tf.int32),
				tf.cast(input_ori_ids, tf.int32)
		)
		sampled_not_equal = tf.cast(sampled_not_equal_id, tf.float32) * tf.cast(input_mask, tf.float32)
		sampled_not_equal = 1 - tf.reduce_sum(sampled_not_equal) / (1e-10 + tf.reduce_sum(tf.cast(label_diff_ids, tf.float32)))
				
		if kargs.get('summary_debug', False):
			tf.summary.scalar('generator_sample_acc', 
							sampled_not_equal)

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

	

			
