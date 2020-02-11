import tensorflow as tf
import numpy as np

from utils.bert import bert_utils
from utils.bert import bert_modules, albert_modules

import tensorflow as tf
from tensorflow.python.framework import ops

def inverse_exp_decay(max_step, min_value=0.01, step=None):
	"""Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
	inv_base = tf.exp(tf.log(min_value) / float(max_step))
	if step is None:
		step = tf.train.get_or_create_global_step()
	if step is None:
		return 1.0
	step = tf.cast(step, tf.float32)
	return inv_base**tf.maximum(float(max_step) - step, 0.0)

def inverse_temp_exp_decay(max_step, max_inv_temp, step=None):
	if step is None:
		step = tf.train.get_or_create_global_step()
	if step is None:
		return 1.0
	step = tf.cast(step, tf.float32)
	return 1.0 / tf.pow(float(max_inv_temp), tf.minimum(step / float(max_step), 1.0))

def inverse_lin_decay(max_step, min_value=0.01, step=None):
	"""Inverse-decay linearly from min_value to 1.0 reached at max_step."""
	if step is None:
		step = tf.train.get_or_create_global_step()
	if step is None:
		return 1.0
	step = tf.cast(step, tf.float32)
	progress = tf.minimum(step / float(max_step), 1.0)
	return progress * (1.0 - min_value) + min_value

class FlipGradientBuilder(object):
	def __init__(self):
		self.num_calls = 0

	def __call__(self, x, l=1.0):
		grad_name = "FlipGradient%d" % self.num_calls
		@ops.RegisterGradient(grad_name)
		def _flip_gradients(op, grad):
			return [tf.negative(grad) * l]
		
		g = tf.get_default_graph()
		with g.gradient_override_map({"Identity": grad_name}):
			y = tf.identity(x)
			
		self.num_calls += 1
		return y
	
flip_gradient = FlipGradientBuilder()

@tf.custom_gradient
def gumbel_softmax_grad(x):
	y = tf.exp(tf.nn.log_softmax(x, axis=1))
	def grad(dy):
		print("===gumbel-softmax gradient===")
		return dy * tf.stop_gradient(y) * tf.gradients(tf.log(y+1e-20), x)[0]
	return y, grad

def gumbel_logsoftmax_grad(x):
	return tf.nn.log_softmax(x, axis=1)

def sample_gumbel(shape, samples=1, eps=1e-20): 
	"""Sample from Gumbel(0, 1)"""
	if samples > 1:
		sample_shape = shape + [samples]
	else:
		sample_shape = shape
	U = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
	# return -tf.log(-tf.log(U + eps) + eps)
	return -tf.log(-tf.log(U))

def gumbel_softmax(logits, temperature, gumbel_samples=None, samples=1, greedy=False): 
	""" Draw a sample from the Gumbel-Softmax distribution"""
	input_shape_list = bert_utils.get_shape_list(logits, expected_rank=2)
	if samples > 1:
		logits = tf.expand_dims(logits, -1)
	if gumbel_samples is None:
		y = logits + sample_gumbel(input_shape_list, samples)
	else:
		y = logits + gumbel_samples
	if greedy:
		tf.logging.info("==apply greedy based sampling and discrete relax==")
		return [tf.exp(tf.nn.log_softmax(logits / temperature, axis=1)),
				logits]
	else:
		tf.logging.info("==apply sampling based sampling and discrete relax==")
		return [tf.exp(tf.nn.log_softmax(y / temperature, axis=1)), 
				y]

def gumbel_softmax_custom_grad(logits, temperature, gumbel_samples=None, samples=1): 
	""" Draw a sample from the Gumbel-Softmax distribution"""
	input_shape_list = bert_utils.get_shape_list(logits, expected_rank=2)
	if samples > 1:
		logits = tf.expand_dims(logits, -1)
	if gumbel_samples is None:
		y = logits + sample_gumbel(input_shape_list, samples)
	else:
		y = logits + gumbel_samples
	return [gumbel_logsoftmax_grad(y / temperature), 
			y]

def token_generator_gumbel(config, input_tensor,
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

		# if config.get("embedding", "factorized") == "factorized":
		# 	projection_width = config.hidden_size
		# else:
		# 	projection_width = config.embedding_size

		if config.get("embedding", "none_factorized") == "none_factorized":
			projection_width = config.hidden_size
			tf.logging.info("==not using embedding factorized==")
		else:
			projection_width = config.get('embedding_size', config.hidden_size)
			tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

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

		# it seems no need for logits to be normalized
		logits_tempered = logits #tf.nn.log_softmax(logits, axis=-1)

		# width=config.vocab_size
		flat_logits_tempered = tf.reshape(logits_tempered,
									[batch_size * seq_length, width])

		num_train_steps = kargs.get('num_train_steps', None)
		if num_train_steps and kargs.get('gumbel_anneal', "anneal") == 'anneal':
			tf.logging.info("****** apply annealed temperature ******* %s", str(num_train_steps))
			temperature_warmup_steps = int(num_train_steps) * 0.1
			annealed_temp = tf.train.polynomial_decay(config.get('gumbel_temperature', 1.0),
													tf.train.get_or_create_global_step(),
													temperature_warmup_steps,
													end_learning_rate=0.01,
													power=1.0,
													cycle=False)
			gumbel_samples = None
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('annealed_temp', 
							annealed_temp)
		elif kargs.get('gumbel_anneal', "anneal") == 'softplus':
			tf.logging.info("****** apply auto-scale temperature *******")
			# batch x seq x dim
			with tf.variable_scope("gumbel_auto_scaling_temperature"):
				annealed_temp = tf.layers.dense(input_tensor, 
												1,
												activation=tf.nn.softplus,
												) + 1.0
				annealed_temp = 1./ annealed_temp
				annealed_temp = tf.reshape(annealed_temp, [batch_size * seq_length, 1])
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('softplus temperature', 
							tf.reduce_mean(annealed_temp))
			if config.get('gen_sample', 1) > 1:
				tf.logging.info("****** apply auto-scale temperature for multi-sampling *******")
				annealed_temp = tf.expand_dims(annealed_temp, -1)
			gumbel_samples = None
		elif kargs.get('gumbel_anneal', 'vqvae') == 'vqvae':
			temperature_warmup_steps = kargs.get("num_train_steps", 10000) * 0.1
			tf.logging.info("****** apply t2t gumbel-softmax temperature annealing method with warm up steps %s ******* ",
							str(kargs.get("num_train_steps", 10000) * 0.1))
			steps = temperature_warmup_steps
			gumbel_samples = sample_gumbel(bert_utils.get_shape_list(flat_logits_tempered, expected_rank=2),
											samples=config.get('gen_sample', 1))
			gumbel_samples *= inverse_exp_decay(steps // 5) * 0.5
			annealed_temp_decay = 1.01 - inverse_lin_decay(steps) # minimum temperature is set 0.2
			annealed_temp = tf.cond(
						tf.less(tf.random_uniform([]), 0.9), lambda: annealed_temp_decay,
						lambda: tf.random_uniform([], minval=0.5, maxval=1.0)) # 10% step for 
			tf.logging.info("****** apply t2t gumbel-softmax temperature annealing method ******* ")
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('t2t_vqvae_stgs temperature', 
							annealed_temp)
				tf.summary.scalar('t2t_vqvae_stgs temperature decay', 
							annealed_temp_decay)
		elif kargs.get('gumbel_anneal', 'vqvae_v1') == 'vqvae_v1':
			temperature_warmup_steps = kargs.get("num_train_steps", 10000) * 0.1
			tf.logging.info("****** apply t2t gumbel-softmax temperature annealing method with warm up steps %s ******* ",
							str(kargs.get("num_train_steps", 10000) * 0.1))
			steps = temperature_warmup_steps
			gumbel_samples = sample_gumbel(bert_utils.get_shape_list(flat_logits_tempered, expected_rank=2),
											samples=config.get('gen_sample', 1))
			# gumbel_samples *= inverse_exp_decay(steps)
			annealed_temp_decay = 1.01 - inverse_exp_decay(kargs.get("num_train_steps", 10000)) # minimum temperature is set 0.2
			annealed_temp = annealed_temp_decay
			# annealed_temp = tf.cond(
			# 			tf.less(tf.random_uniform([]), 0.95), lambda: annealed_temp_decay,
			# 			lambda: tf.random_uniform([], minval=0.5, maxval=1.0)) # 10% step for 
			tf.logging.info("****** apply sel-gan gumbel-softmax temperature annealing method ******* ")
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('t2t_vqvae_stgs temperature', 
							annealed_temp)
				tf.summary.scalar('t2t_vqvae_stgs temperature decay', 
							annealed_temp_decay)
		elif kargs.get('gumbel_anneal', 'vqvae_v2') == 'vqvae_v2':
			temperature_warmup_steps = kargs.get("num_train_steps", 10000) * 0.1
			tf.logging.info("****** apply t2t gumbel-softmax temperature annealing method with warm up steps %s ******* ",
							str(kargs.get("num_train_steps", 10000) * 0.1))
			steps = temperature_warmup_steps
			gumbel_samples = sample_gumbel(bert_utils.get_shape_list(flat_logits_tempered, expected_rank=2),
											samples=config.get('gen_sample', 1))
			# gumbel_samples *= inverse_exp_decay(steps)
			annealed_temp_decay = inverse_temp_exp_decay(kargs.get("num_train_steps", 10000), kargs.get("max_temp", 100))
			annealed_temp = annealed_temp_decay
			# annealed_temp = tf.cond(
			# 			tf.less(tf.random_uniform([]), 0.95), lambda: annealed_temp_decay,
			# 			lambda: tf.random_uniform([], minval=0.5, maxval=1.0)) # 10% step for 
			tf.logging.info("****** apply sel-gan-v2 gumbel-softmax temperature annealing method ******* ")
			tf.logging.info("****** apply sel-gan-v2 gumbel-softmax num_train_steps:%s annealing method, temp:%s ******* ", str(kargs.get("num_train_steps", 10000)), str(kargs.get("max_temp", 100)))
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('t2t_vqvae_stgs temperature', 
							annealed_temp)
				tf.summary.scalar('t2t_vqvae_stgs temperature decay', 
							annealed_temp_decay)
		else:
			annealed_temp = 0.01
			gumbel_samples = None
			tf.logging.info("****** not apply annealed tenperature with fixed temp ******* %s", str(annealed_temp))
			if not kargs.get('use_tpu', True):
				tf.summary.scalar('gumbel_temperature', 
							annealed_temp)

		# [batch x seq] x config.vocab_size x config.get('gen_sample', 1)
		if kargs.get('stable_gradient', True):
			sampled_logprob_temp, sampled_logprob = gumbel_softmax(flat_logits_tempered, 
										temperature=annealed_temp,
										gumbel_samples=gumbel_samples,
										samples=config.get('gen_sample', 1),
										greedy=kargs.get("greedy", True))
			tf.logging.info("****** apply normal derivate for gradient calculation *******")
		else:
			sampled_logprob_temp, sampled_logprob = gumbel_softmax_custom_grad(flat_logits_tempered, 
										temperature=annealed_temp,
										gumbel_samples=gumbel_samples,
										samples=config.get('gen_sample', 1))
			tf.logging.info("****** apply log deriviate for stable gradient calculation *******")

		# argmax on config.vocab_size which is always axis=1
		# [batch x seq] x config.vocab_size x config.get('gen_sample', 1)
		# armax(logits+gumbel_samples) to sample a categoritical distribution
		if kargs.get('sampled_prob_id', False):
			tf.logging.info("****** apply categorical sampled id of original logits *******")
			sampled_hard_id = tf.one_hot(tf.argmax(sampled_logprob, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id
		else:
			tf.logging.info("****** apply gumbel-softmax logprob for logits *******")
			sampled_hard_id = tf.one_hot(tf.argmax(sampled_logprob_temp, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id

		# straight-through gumbel softmax estimator
		if kargs.get('if_flip_grad', True):
			tf.logging.info("****** apply gradient flipping *******")
			sampled_logprob_temp_1 = flip_gradient(sampled_logprob_temp)
		else:
			tf.logging.info("****** not apply gradient flipping *******")
			sampled_logprob_temp_1 = sampled_logprob_temp
		if kargs.get("straight_through", False):
			tf.logging.info("****** apply straight_through_estimator *******")
			sampled_id = tf.stop_gradient(sampled_hard_id-sampled_logprob_temp) + (sampled_logprob_temp_1)
		else:
			tf.logging.info("****** apply gumbel-softmax probs *******")
			sampled_id = sampled_logprob_temp_1

		sampled_binary_mask = kargs.get('sampled_binary_mask', None)
		if sampled_binary_mask is not None:
			label_diff_ids =  tf.identity(sampled_binary_mask) # 0 for original and 1 for replace
		else:
			label_diff_ids = tf.not_equal(
							tf.cast(input_ids, tf.int32),
							tf.cast(input_ori_ids, tf.int32) # 0 for original and 1 for replace
						)

		label_diff_ids = tf.cast(label_diff_ids, tf.float32)

		label_diff_ids = tf.expand_dims(label_diff_ids, axis=[-1]) # batch x seq x 1
		input_ori_ids_1 = input_ori_ids
		input_ori_ids = tf.one_hot(input_ori_ids, config.vocab_size) # batch x seq x vocab
		input_ori_ids = tf.cast(input_ori_ids, tf.float32)

		if config.get('gen_sample', 1) == 1:
			sampled_input_id = tf.reshape(sampled_id, [batch_size, seq_length, config.vocab_size])
			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				label_diff_ids = tf.cast(label_diff_ids, tf.float32)
				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - label_diff_ids) * tf.cast(input_ori_ids, tf.float32)
			elif kargs.get('mask_method', 'only_mask') == 'all_mask':
				unk_mask = tf.cast(tf.math.equal(input_ori_ids_1, 100), tf.float32) # not replace unk
				cls_mask =  tf.cast(tf.math.equal(input_ori_ids_1, 101), tf.float32) # not replace cls
				sep_mask = tf.cast(tf.math.equal(input_ori_ids_1, 102), tf.float32) # not replace sep
				unsampled_mask = (1 - (unk_mask + cls_mask + sep_mask))*tf.cast(input_mask, tf.float32)
				unsampled_mask = tf.expand_dims(unsampled_mask, axis=[-1]) # batch x seq x 1
				tf.logging.info("****** all mask sample *******")
				sampled_input_id = unsampled_mask * tf.cast(sampled_input_id, tf.float32) + (1 - unsampled_mask) * tf.cast(input_ori_ids, tf.float32)
		else:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length, config.vocab_size, config.get('gen_sample', 1)])
			label_diff_ids = tf.expand_dims(label_diff_ids, axis=-1) # batch x seq x 1
			input_ori_ids = tf.expand_dims(input_ori_ids, axis=-1) # batch x seq x vocab x 1

			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - input_ori_ids) * label_diff_ids

		tf.logging.info("====generator use_tpu %s ====", str(kargs.get('use_tpu', True)))
		if not kargs.get('use_tpu', True):
			tf.logging.info("====logging generator loss ====")
			sampled_not_equal_id = tf.not_equal(
				tf.cast(tf.argmax(sampled_input_id, axis=2), tf.int32),
				tf.cast(tf.argmax(input_ori_ids, axis=2), tf.int32)
			)
			sampled_not_equal = tf.cast(sampled_not_equal_id, tf.float32) * tf.cast(input_mask, tf.float32)
			sampled_not_equal = 1 - tf.reduce_sum(sampled_not_equal) / (1e-10 + tf.reduce_sum(tf.cast(label_diff_ids, tf.float32)))
			tf.summary.scalar('generator_sample_acc', 
							sampled_not_equal)

			sampled_hard_id = tf.one_hot(tf.argmax(sampled_logprob_temp, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id
			sampled_hard_id = tf.cast(sampled_hard_id, tf.float32)
			sampled_hard_id = tf.reshape(sampled_hard_id, [batch_size, seq_length, config.vocab_size])
			label_diff_ids_my = tf.cast(label_diff_ids, tf.float32)
			sampled_soft_id = tf.reshape(sampled_id, [batch_size, seq_length, config.vocab_size])
			sampled_hard_id *= label_diff_ids_my
			sampled_soft_id *= label_diff_ids_my

			hard_soft_bias = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(sampled_hard_id - sampled_soft_id, 2), axis=-1))) / (1e-10+tf.reduce_sum(tf.cast(label_diff_ids_my, tf.float32)))
			tf.summary.scalar('soft_hard_bias', 
							hard_soft_bias)

		return sampled_input_id

def token_generator_gumbel_normal(config, input_tensor,
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

		# if config.get("embedding", "factorized") == "factorized":
		# 	projection_width = config.hidden_size
		# else:
		# 	projection_width = config.embedding_size

		if config.get("embedding", "none_factorized") == "none_factorized":
			projection_width = config.hidden_size
			tf.logging.info("==not using embedding factorized==")
		else:
			projection_width = config.get('embedding_size', config.hidden_size)
			tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

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

		logits_tempered = tf.nn.log_softmax(logits, axis=-1)

		# width=config.vocab_size
		flat_logits_tempered = tf.reshape(logits_tempered,
									[batch_size * seq_length, width])

		num_train_steps = kargs.get('num_train_steps', None)
		if num_train_steps and kargs.get('gumbel_anneal', "anneal") == 'anneal':
			tf.logging.info("****** apply annealed temperature ******* %s", str(num_train_steps))
			annealed_temp = tf.train.polynomial_decay(config.get('gumbel_temperature', 1.0),
													tf.train.get_or_create_global_step(),
													kargs.get("num_train_steps", 10000),
													end_learning_rate=0.1,
													power=1.0,
													cycle=False)
		elif kargs.get('gumbel_anneal', "anneal") == 'softplus':
			tf.logging.info("****** apply auto-scale temperature *******")
			# batch x seq x dim
			with tf.variable_scope("gumbel_auto_scaling_temperature"):
				annealed_temp = tf.layers.dense(input_tensor, 
												1,
												activation=tf.nn.softplus,
												) + 1.0
				annealed_temp = 1./ annealed_temp
				annealed_temp = tf.reshape(annealed_temp, [batch_size * seq_length, 1])
			if config.get('gen_sample', 1) > 1:
				tf.logging.info("****** apply auto-scale temperature for multi-sampling *******")
				annealed_temp = tf.expand_dims(annealed_temp, -1)
		else:
			annealed_temp = 1.0
			tf.logging.info("****** not apply annealed tenperature with fixed temp ******* %s", str(annealed_temp))
			
		# [batch x seq] x config.vocab_size x config.get('gen_sample', 1)
		sampled_logprob_temp, sampled_logprob = gumbel_softmax(flat_logits_tempered, 
										temperature=annealed_temp,
										samples=config.get('gen_sample', 1))

		# argmax on config.vocab_size which is always axis=1
		# [batch x seq] x config.vocab_size x config.get('gen_sample', 1)
		# armax(logits+gumbel_samples) to sample a categoritical distribution
		if kargs.get('sampled_prob_id', True):
			tf.logging.info("****** apply categorical sampled id of original logits *******")
			sampled_hard_id = tf.one_hot(tf.argmax(sampled_logprob, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id
		else:
			tf.logging.info("****** apply gumbel-softmax logprob for logits *******")
			sampled_hard_id = tf.one_hot(tf.argmax(sampled_logprob_temp, axis=1), 
									config.vocab_size,
									axis=1) # sampled multiminal id

		# straight-through gumbel softmax estimator
		if kargs.get("straight_through", True):
			tf.logging.info("****** apply straight_through_estimator without grl *******")
			sampled_id = tf.stop_gradient(sampled_hard_id-sampled_logprob_temp) + (sampled_logprob_temp)
		else:
			tf.logging.info("****** apply gumbel-softmax probs without grl *******")
			sampled_id = flip_gradient(sampled_logprob_temp)

		sampled_binary_mask = kargs.get('sampled_binary_mask', None)
		if sampled_binary_mask is not None:
			label_diff_ids =  tf.identity(sampled_binary_mask) # 0 for original and 1 for replace
		else:
			label_diff_ids = tf.not_equal(
							tf.cast(input_ids, tf.int32),
							tf.cast(input_ori_ids, tf.int32) # 0 for original and 1 for replace
						)

		label_diff_ids = tf.cast(label_diff_ids, tf.float32)

		label_diff_ids = tf.expand_dims(label_diff_ids, axis=[-1]) # batch x seq x 1
		input_ori_ids = tf.one_hot(input_ori_ids, config.vocab_size) # batch x seq x vocab
		input_ori_ids = tf.cast(input_ori_ids, tf.float32)

		if config.get('gen_sample', 1) == 1:
			sampled_input_id = tf.reshape(sampled_id, [batch_size, seq_length, config.vocab_size])
			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				label_diff_ids = tf.cast(label_diff_ids, tf.float32)
				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - label_diff_ids) * tf.cast(input_ori_ids, tf.float32)
		else:
			sampled_input_id = tf.reshape(samples, [batch_size, seq_length, config.vocab_size, config.get('gen_sample', 1)])
			label_diff_ids = tf.expand_dims(label_diff_ids, axis=-1) # batch x seq x 1
			input_ori_ids = tf.expand_dims(input_ori_ids, axis=-1) # batch x seq x vocab x 1

			if kargs.get('mask_method', 'only_mask') == 'only_mask':
				tf.logging.info("****** only mask sample *******")
				sampled_input_id = (label_diff_ids) * tf.cast(sampled_input_id, tf.float32) + (1 - input_ori_ids) * label_diff_ids
				
		tf.logging.info("====generator use_tpu %s ====", str(kargs.get('use_tpu', True)))
		if not kargs.get('use_tpu', True):
			tf.logging.info("====logging generator loss ====")
			sampled_not_equal_id = tf.not_equal(
				tf.cast(tf.argmax(sampled_input_id, axis=2), tf.int32),
				tf.cast(tf.argmax(input_ori_ids, axis=2), tf.int32)
			)
			sampled_not_equal = tf.cast(sampled_not_equal_id, tf.float32) * tf.cast(input_mask, tf.float32)
			sampled_not_equal = 1 - tf.reduce_sum(sampled_not_equal) / (1e-10 + tf.reduce_sum(tf.cast(label_diff_ids, tf.float32)))
			tf.summary.scalar('generator_sample_acc', 
							sampled_not_equal)

		return sampled_input_id








