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

import copy

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

	ngram_list = kargs.get("ngram", [10, 3])
	mask_prob_list = kargs.get("mask_prob", [0.2, 0.2])
	ngram_ratio = kargs.get("ngram_ratio", [8, 1])
	uniform_ratio = kargs.get("uniform_ratio", 1.0)
	tf.logging.info("****** dynamic ngram: %s, mask_prob: %s, mask_prior: %s, uniform_ratio: %s *******", 
			str(ngram_list), str(mask_prob_list), str(ngram_ratio), str(uniform_ratio))	
	tran_prob_list, hmm_tran_prob_list = [], []
	for ngram_sub, mask_prob_sub in zip(ngram_list, mask_prob_list):
		tran_prob, hmm_tran_prob = ngram_prob(ngram_sub, mask_prob_sub)
		tran_prob_list.append(tran_prob)
		hmm_tran_prob_list.append(hmm_tran_prob)
	mask_prior = []
	for ratio in ngram_ratio:
		actual_ratio = (1 - uniform_ratio) / sum(ngram_ratio) * ratio
		mask_prior.append(actual_ratio)
	mask_prior.append(uniform_ratio)
	tf.logging.info("****** mask prior: %s *******", str(mask_prior))
	mask_prior = np.array(mask_prior).astype(np.float32)
	
	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)
		[output_ids, 
		sampled_binary_mask] = hmm_input_ids_generation(model_config,
									features['input_ori_ids'],
									features['input_mask'],
									[tf.cast(tf.constant(hmm_tran_prob), tf.float32) for hmm_tran_prob in hmm_tran_prob_list],
									mask_probability=0.3,
									replace_probability=0.1,
									original_probability=0.1,
									mask_prior=tf.constant(mask_prior, tf.float32),
									**kargs)

		features['input_ids'] = output_ids
				
		model = model_api(model_config, features, labels,
							tf.estimator.ModeKeys.EVAL, target, reuse=tf.AUTO_REUSE,
							**kargs)

		sampled_ids = token_generator(model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(), 
									features['input_ids'], 
									features['input_ori_ids'],
									features['input_mask'],	
									embedding_projection=model.get_embedding_projection_table(),
									scope=generator_scope_prefix,
									mask_method='only_mask',
									use_tpu=kargs.get('use_tpu', True),
									apply_valid_vocab=kargs.get('apply_valid_vocab', True),
									invalid_size=kargs.get('invalid_size', 106))

		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		if generator_scope_prefix:
			"""
			"generator/cls/predictions"
			"""
			lm_pretrain_tvars = model_io_fn.get_params(generator_scope_prefix+"/cls/predictions", 
										not_storage_params=not_storage_params)

			nsp_pretrain_vars = model_io_fn.get_params(generator_scope_prefix+"/cls/seq_relationship",
										not_storage_params=not_storage_params)
		else:
			lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
										not_storage_params=not_storage_params)

			nsp_pretrain_vars = model_io_fn.get_params("cls/seq_relationship",
										not_storage_params=not_storage_params)

		if model_config.get('embedding_scope', None) is not None:
			embedding_tvars = model_io_fn.get_params(model_config.get('embedding_scope', 'bert')+"/embeddings", 
									not_storage_params=not_storage_params)
			pretrained_tvars.extend(embedding_tvars)

		pretrained_tvars.extend(lm_pretrain_tvars)
		pretrained_tvars.extend(nsp_pretrain_vars)
		tvars = pretrained_tvars

		print('==generator parameters==', tvars)

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu,
											restore_var_name=model_config.get('restore_var_name', []))
		else:
			scaffold_fn = None

		return_dict = {
					"tvars":tvars,
					"sampled_ids":sampled_ids
				}
		return return_dict
	return model_fn
		
