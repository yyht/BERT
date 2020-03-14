import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

from pretrain_finetuning.token_generator import token_generator, random_input_ids_generation
from pretrain_finetuning.token_generator_hmm import hmm_input_ids_generation, ngram_prob

from utils.bert import bert_utils
from model_io import model_io
from utils.bert import bert_seq_utils, bert_seq_tpu_utils
import copy

def get_fixed_temperature(temper, i, nadv_steps, adapt):
	# using a fixed number of maximum adversarial steps
	N = float(nadv_steps)
	temper = float(temper)
	if adapt == 'no':
		temper_var_np = temper  # no increase
	elif adapt == 'lin':
		temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
	elif adapt == 'exp':
		temper_var_np = temper ** (i / N)  # exponential increase
	elif adapt == 'log':
		temper_var_np = 1 + (temper - 1) / tf.log(N) * tf.log(i + 1)  # logarithm increase
	elif adapt == 'sigmoid':
		temper_var_np = (temper - 1) * 1 / (1 + tf.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
	elif adapt == 'quad':
		temper_var_np = (temper - 1) / (N - 1)**2 * i ** 2 + 1
	elif adapt == 'sqrt':
		temper_var_np = (temper - 1) / tf.sqrt(N - 1) * tf.sqrt(i) + 1
	else:
		raise Exception("Unknown adapt type!")

	return temper_var_np

def model_fn_builder(
					model_config,
					num_labels,
					init_checkpoint,
					model_reuse=None,
					load_pretrained=True,
					model_io_config={},
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="",
					**kargs):

	model_config = copy.deepcopy(model_config)
	if kargs.get("sharing_mode", "none") == "none":
		"""
		'generator/' + model_config.scope
		"""
		model_config.scope = exclude_scope + '/' + model_config.scope
		generator_scope_prefix = exclude_scope
		exclude_scope = exclude_scope
		tf.logging.info("****** generator parameter *******")
	elif kargs.get("sharing_mode", "none") == "all_sharing":
		generator_scope_prefix = None
		exclude_scope = ''
		tf.logging.info("****** generator parameter sharing with discriminator *******")

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		return_dict = {}

		if kargs.get("noise_true_distribution", True):
			model = model_api(model_config, features, labels,
								mode, target, reuse=tf.AUTO_REUSE,
								scope=generator_scope_prefix, # need to add noise scope to lm
								**kargs)

			sequence_mask = tf.to_float(tf.not_equal(features['input_ids'][:, 1:], 
														kargs.get('[PAD]', 0)))

			# batch x seq_length
			seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=features['input_ids'][:, 1:], 
						logits=model.get_sequence_output_logits()[:, :-1])

			if not kargs.get("prob_ln", False):
				tf.logging.info("****** sum of plogprob as sentence probability of noise true data *******")
				logits = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) #/ (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
			else:
				tf.logging.info("****** sum of plogprob with length normalization as sentence probability of noise true data *******")
				logits = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
			# since sparse_softmax_cross_entropy_with_logits will output -logits for minimization
			# while we actually need the log_prob, so we need to minus logits
			return_dict['true_logits'] = -logits
			return_dict['true_seq_logits'] = model.get_sequence_output_logits()
			tf.logging.info("****** noise distribution for true data *******")

		noise_estimator_type = kargs.get("noise_estimator_type", "straight_through")
		tf.logging.info("****** noise estimator for nce: %s *******", noise_estimator_type)

		# with tf.variable_scope("noise", reuse=tf.AUTO_REUSE):
		# 	noise_global_step = tf.get_variable(
		# 						"global_step",
		# 						shape=[],
		# 						initializer=tf.constant_initializer(0, dtype=tf.int64),
		# 						trainable=False,
		# 						dtype=tf.int64)
		# return_dict['global_step'] = noise_global_step

		if kargs.get("sample_noise_dist", True):
			tf.logging.info("****** noise distribution for fake data *******")

			temp_adapt = kargs.get("gumbel_adapt", "exp")
			temper = kargs.get("gumbel_inv_temper", 100)
			num_train_steps = kargs.get("num_train_steps", 100000)

			# step = tf.cast(return_dict['global_step'], tf.float32)
			step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

			temperature = get_fixed_temperature(temper, step, num_train_steps, temp_adapt)

			sample_type = kargs.get("sample_type", "cache_sample")

			if sample_type == 'none_cache_sample':
				sample_sequence_api = bert_seq_tpu_utils.sample_sequence_without_cache
				if_bp = False
				if_cache_decode = False
				tf.logging.info("****** noise sample without cache *******")
			elif sample_type == 'cache_sample':
				sample_sequence_api = bert_seq_tpu_utils.sample_sequence
				if_bp = True
				if_cache_decode = True
				tf.logging.info("****** noise sample with cache *******")
			else:
				sample_sequence_api = bert_seq_tpu_utils.sample_sequence_without_cache
				if_bp = False
				if_cache_decode = False
				tf.logging.info("****** noise sample without cache *******")
			tf.logging.info("****** max_length: %s *******", str(kargs.get('max_length', 512)))

			if noise_estimator_type in ["straight_through", "soft"]:
				back_prop = True
				tf.logging.info("****** st or soft with bp: %s *******", str(back_prop))
			else:
				back_prop = False
				tf.logging.info("****** hard without bp: %s *******", str(back_prop))

			results = sample_sequence_api(model_api,
											model_config, 
											tf.estimator.ModeKeys.PREDICT, 
											features,
											target="", 
											start_token=kargs.get("start_token_id", 101), 
											batch_size=None, 
											context=None, #features["input_ids"][:, :32], 
											temperature=1.0, 
											n_samples=kargs.get("n_samples", 1),
											top_k=0,
											end_token=kargs.get("end_token_id", 102),
											greedy_or_sample="sample",
											gumbel_temp=temperature,
											estimator=noise_estimator_type,
											back_prop=back_prop,
											swap_memory=True,
											seq_type=kargs.get("seq_type", "seq2seq"),
											mask_type=kargs.get("mask_type", "left2right"),
											attention_type=kargs.get('attention_type', 'normal_attention'),
											scope=generator_scope_prefix, # need to add noise scope to lm,
											max_length=max(int(kargs.get('max_length', 512)/6), 32),
											if_bp=if_bp,
											if_cache_decode=if_cache_decode
											)

			if noise_estimator_type in ["straight_through", "soft"]:
				tf.logging.info("****** using apply gumbel samples *******")
				gumbel_probs = results['gumbel_probs']
			else:
				gumbel_probs = tf.cast(results['samples'], tf.int32)
				tf.logging.info("****** using apply stop gradient samples *******")
			return_dict['gumbel_probs'] = tf.cast(gumbel_probs, tf.float32)
			sample_mask = results['mask_sequence']
			if not kargs.get("prob_ln", False):
				tf.logging.info("****** sum of plogprob as sentence probability of noise sampled data *******")
				return_dict['fake_logits'] = tf.reduce_sum(results['logits']*tf.cast(sample_mask, tf.float32), axis=-1) #/ tf.reduce_sum(1e-10+tf.cast(sample_mask, tf.float32), axis=-1)
			else:
				tf.logging.info("****** sum of plogprob with length normalization as sentence probability of noise sampled data *******")
				return_dict['fake_logits'] = tf.reduce_sum(results['logits']*tf.cast(sample_mask, tf.float32), axis=-1) / tf.reduce_sum(1e-10+tf.cast(sample_mask, tf.float32), axis=-1)
			return_dict['fake_samples'] = tf.cast(results['samples'], tf.int32)
			return_dict['fake_mask'] = results['mask_sequence']

			print(return_dict['fake_samples'].get_shape(), return_dict['fake_logits'].get_shape(), results['logits'].get_shape(),  "====fake samples, logitss, shape===")
			
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		if generator_scope_prefix:
			"""
			"generator/cls/predictions"
			"""
			lm_pretrain_tvars = model_io_fn.get_params(generator_scope_prefix+"/cls/predictions", 
										not_storage_params=not_storage_params)
		else:
			lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
										not_storage_params=not_storage_params)

		if model_config.get('embedding_scope', None) is not None:
			embedding_tvars = model_io_fn.get_params(model_config.get('embedding_scope', 'bert')+"/embeddings", 
									not_storage_params=not_storage_params)
			pretrained_tvars.extend(embedding_tvars)

		pretrained_tvars.extend(lm_pretrain_tvars)
		return_dict['tvars'] = pretrained_tvars

		use_tpu = 1 if kargs.get('use_tpu', False) else 0

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu)
		else:
			scaffold_fn = None

		return return_dict
	return model_fn
