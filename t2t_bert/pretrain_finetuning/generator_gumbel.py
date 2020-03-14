import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

from pretrain_finetuning.token_generator import random_input_ids_generation
from pretrain_finetuning.token_generator_gumbel import token_generator_gumbel
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

	model_config.hidden_size = int(round(
	  model_config.hidden_size * model_config.get('generator_hidden_size', 1.0)))
	model_config.num_hidden_layers = int(round(
	  model_config.num_hidden_layers * model_config.get('layer_ratio', 1.0)))
	model_config.intermediate_size = 4 * model_config.hidden_size
	model_config.num_attention_heads = max(1, model_config.hidden_size // 64)

	print(model_config, "==generator config==")

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

		if kargs.get('random_generator', '1') == '1':
			if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
				input_ori_ids = features['input_ori_ids']

				# [output_ids, 
				# sampled_binary_mask] = random_input_ids_generation(model_config,
				# 							features['input_ori_ids'],
				# 							features['input_mask'],
				# 							mask_probability=0.2,
				# 							replace_probability=0.1,
				# 							original_probability=0.1,
				# 							**kargs)

				[output_ids, 
				sampled_binary_mask] = hmm_input_ids_generation(model_config,
											features['input_ori_ids'],
											features['input_mask'],
											[tf.cast(tf.constant(hmm_tran_prob), tf.float32) for hmm_tran_prob in hmm_tran_prob_list],
											mask_probability=0.2,
											replace_probability=0.1,
											original_probability=0.1,
											mask_prior=tf.cast(tf.constant(mask_prior), tf.float32),
											**kargs)

				features['input_ids'] = tf.identity(output_ids)

				tf.logging.info("****** do random generator *******")
			else:
				sampled_binary_mask = None
				output_ids = tf.identity(features['input_ids'])
		else:
			sampled_binary_mask = None
			output_ids = tf.identity(features['input_ids'])

		model = model_api(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope
		
		(nsp_loss, 
		 nsp_per_example_loss, 
		 nsp_log_prob) = pretrain.get_next_sentence_output(model_config,
										model.get_pooled_output(),
										features['next_sentence_labels'],
										reuse=tf.AUTO_REUSE,
										scope=generator_scope_prefix)

		masked_lm_positions = features["masked_lm_positions"]
		masked_lm_ids = features["masked_lm_ids"]
		masked_lm_weights = features["masked_lm_weights"]

		if model_config.model_type == 'bert':
			masked_lm_fn = pretrain.get_masked_lm_output
			seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
			print("==apply bert masked lm==")
		elif model_config.model_type == 'albert':
			masked_lm_fn = pretrain_albert.get_masked_lm_output
			seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
			print("==apply albert masked lm==")
		else:
			masked_lm_fn = pretrain.get_masked_lm_output
			seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
			print("==apply bert masked lm==")

		if sampled_binary_mask is not None:
			(masked_lm_loss,
			masked_lm_example_loss, 
			masked_lm_log_probs,
			masked_lm_mask) = seq_masked_lm_fn(model_config, 
										model.get_sequence_output(), 
										model.get_embedding_table(),
										features['input_mask'], 
										features['input_ori_ids'], 
										features['input_ids'],
										sampled_binary_mask,
										reuse=tf.AUTO_REUSE,
										embedding_projection=model.get_embedding_projection_table(),
										scope=generator_scope_prefix)
			masked_lm_ids = features['input_ori_ids']
		else:
			(masked_lm_loss,
			masked_lm_example_loss, 
			masked_lm_log_probs,
			masked_lm_mask) = masked_lm_fn(
											model_config, 
											model.get_sequence_output(), 
											model.get_embedding_table(),
											masked_lm_positions, 
											masked_lm_ids, 
											masked_lm_weights,
											reuse=tf.AUTO_REUSE,
											embedding_projection=model.get_embedding_projection_table(),
											scope=generator_scope_prefix)
		print(model_config.lm_ratio, '==mlm lm_ratio==')
		loss = model_config.lm_ratio * masked_lm_loss + 0.0 * nsp_loss

		if kargs.get("resample_discriminator", False):
			input_ori_ids = features['input_ori_ids']

			[output_ids, 
			sampled_binary_mask] = random_input_ids_generation(model_config,
										features['input_ori_ids'],
										features['input_mask'],
										mask_probability=0.2,
										replace_probability=0.1,
										original_probability=0.1)

			resample_features = {}
			for key in features:
				resample_features[key] = features[key]

			resample_features['input_ids'] = tf.identity(output_ids)
			model_resample = model_api(model_config, resample_features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)

			gumbel_model = model_resample
			gumbel_features = resample_features
			tf.logging.info("**** apply discriminator resample **** ")
		else:
			gumbel_model = model
			gumbel_features = features
			tf.logging.info("**** not apply discriminator resample **** ")

		sampled_ids = token_generator_gumbel(model_config, 
										gumbel_model.get_sequence_output(), 
										gumbel_model.get_embedding_table(), 
										gumbel_features['input_ids'], 
										gumbel_features['input_ori_ids'],
										gumbel_features['input_mask'],	
										embedding_projection=gumbel_model.get_embedding_projection_table(),
										scope=generator_scope_prefix,
										# mask_method='only_mask',
										**kargs)

		if model_config.get('gen_sample', 1) == 1:
			input_ids = features['input_ori_ids']
			input_mask = features['input_mask']
			segment_ids = features['segment_ids']

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

		# embedding_tvars = model_io_fn.get_params(model_config.get('embedding_scope', 'bert')+"/embeddings", 
		# 							not_storage_params=not_storage_params)

		# pretrained_tvars.extend(lm_pretrain_tvars)
		# pretrained_tvars.extend(embedding_tvars)
		# tvars = pretrained_tvars

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

		# tf.add_to_collection("generator_loss", masked_lm_loss)
		return_dict = {
					"loss":loss, 
					"tvars":tvars,
					"model":model,
					"sampled_ids":sampled_ids,  # batch x gen_sample, seg_length
					"sampled_input_ids":input_ids,       # batch x gen_sample, seg_length,
					"sampled_input_mask":input_mask,
					"sampled_segment_ids":segment_ids,
					"masked_lm_ids":masked_lm_ids,
					"masked_lm_weights":masked_lm_mask,
					"masked_lm_log_probs":masked_lm_log_probs,
					"masked_lm_example_loss":masked_lm_example_loss,
					"next_sentence_example_loss":nsp_per_example_loss,
					"next_sentence_log_probs":nsp_log_prob, 
					"next_sentence_labels":features['next_sentence_labels'],
					"output_ids":output_ids,
					"sampled_binary_mask":sampled_binary_mask
				}
		return return_dict
	return model_fn
		
