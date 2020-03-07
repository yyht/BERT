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
from utils.bert import bert_seq_utils
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
					target="a",
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

		if mode == tf.estimator.ModeKeys.TRAIN:

			if kargs.get("noise_true_distribution", True):
				model = model_api(model_config, features, labels,
									mode, target, reuse=tf.AUTO_REUSE,
									**kargs)


				sequence_mask = tf.to_float(tf.not_equal(features['input_ids'][:, 1:], 
															kargs.get('[PAD]', 0)))

				# batch x seq_length
				seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
							labels=features['input_ids'][:, 1:], 
							logits=model.get_sequence_output_logits()[:, :-1])

				logits = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
				# since sparse_softmax_cross_entropy_with_logits will output -logits for minimization
				# while we actually need the log_prob, so we need to minus logits
				return_dict['true_logits'] = -logits
				tf.logging.info("****** noise distribution for true data *******")

		noise_estimator_type = kargs.get("noise_estimator_type", "straight_through")
		tf.logging.info("****** noise estimator for nce: %s *******", noise_estimator_type)

		if kargs.get("sample_noise_dist", True):
			tf.logging.info("****** noise distribution for fake data *******")

			temp_adapt = kargs.get("gumbel_adapt", "exp")
			temper = kargs.get("gumbel_inv_temper", 100)
			num_train_steps = kargs.get("num_train_steps", 100000)

			step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

			temperature = get_fixed_temperature(temper, step, num_train_steps, temp_adapt)

			results = bert_seq_utils.sample_sequence(model_api,
											model_config, 
											mode, 
											features,
											target="", 
											start_token=kargs.get("start_token_id", 101), 
											batch_size=None, 
											context=features.get("context", None), 
											temperature=temperature, 
											n_samples=kargs.get("n_samples", 1),
											top_k=0,
											end_token=kargs.get("end_token_id", 102),
											greedy_or_sample="sample",
											gumbel_temp=0.01,
											estimator=noise_estimator_type,
											back_prop=True,
											swap_memory=True,
											seq_type=kargs.get("seq_type", "seq2seq"),
											mask_type=kargs.get("mask_type", "seq2seq"),
											attention_type=kargs.get('attention_type', 'normal_attention')
											)

			if noise_estimator in ["straight_through", "soft"]:
				gumbel_probs = results['gumbel_probs']
			else:
				gumbel_probs = tf.cast(results['samples'], tf.int32)
			return_dict['gumbel_probs'] = tf.cast(gumbel_probs, tf.float32)
			sample_mask = results['mask_sequence']
			return_dict['fake_logits'] = tf.reduce_sum(results['logits']*tf.cast(sample_mask, tf.float32), axis=-1) / tf.reduce_sum(1e-10+tf.cast(sample_mask, tf.float32), axis=-1)
			return_dict['fake_samples'] = tf.cast(results['samples'], tf.int32)
			
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

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