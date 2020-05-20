import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

import tensorflow as tf
import numpy as np
from model_io import model_io
from pretrain_finetuning.token_generator_hmm import hmm_input_ids_generation, ngram_prob
from task_module import classifier
from task_module import tsa_pretrain
import tensorflow as tf
from metric import tf_metrics

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
	model_config.tsa = 'exp_schedule'
	model_config.num_train_steps = opt_config.num_train_steps
	# opt_config.init_lr /= 2

	ngram_list = kargs.get("ngram", [10, 5, 2])
	mask_prob_list = kargs.get("mask_prob", [0.2, 0.2, 0.2])
	ngram_ratio = kargs.get("ngram_ratio", [6, 1, 1])
	uniform_ratio = kargs.get("uniform_ratio", 0.1)
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
	mask_prior = np.array(mask_prior).astype(np.float32)

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)

		input_mask = tf.cast(tf.not_equal(features['input_ids_{}'.format(target)], 
							kargs.get('[PAD]', 0)), tf.int32)
		segment_ids = tf.zeros_like(input_mask)

		if target:
			features['input_ori_ids'] = features['input_ids_{}'.format(target)]
			features['input_mask'] = input_mask
			features['segment_ids'] = segment_ids
			# features['input_mask'] = features['input_mask_{}'.format(target)]
			# features['segment_ids'] = features['segment_ids_{}'.format(target)]
			features['input_ids'] = features['input_ids_{}'.format(target)]

		input_ori_ids = features.get('input_ori_ids', None)
		if mode == tf.estimator.ModeKeys.TRAIN:
			if input_ori_ids is not None:

				[output_ids, 
				sampled_binary_mask] = hmm_input_ids_generation(model_config,
											features['input_ori_ids'],
											features['input_mask'],
											[tf.cast(tf.constant(hmm_tran_prob), tf.float32) for hmm_tran_prob in hmm_tran_prob_list],
											mask_probability=0.1,
											replace_probability=0.1,
											original_probability=0.1,
											mask_prior=tf.cast(tf.constant(mask_prior), tf.float32),
											**kargs)

				features['input_ids'] = output_ids
				tf.logging.info("***** Running random sample input generation *****")
			else:
				sampled_binary_mask = None
		else:
			sampled_binary_mask = None

		model = model_api(model_config, features, labels,
							mode, "", reuse=tf.AUTO_REUSE,
							**kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

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
										embedding_projection=model.get_embedding_projection_table())
			masked_lm_ids = input_ori_ids
		else:
			masked_lm_positions = features["masked_lm_positions"]
			masked_lm_ids = features["masked_lm_ids"]
			masked_lm_weights = features["masked_lm_weights"]
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
											embedding_projection=model.get_embedding_projection_table())
		print(model_config.lm_ratio, '==mlm lm_ratio==')
		loss = model_config.lm_ratio * masked_lm_loss #+ 0.0 * nsp_loss
		
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

		pretrained_tvars.extend(lm_pretrain_tvars)
		tvars = pretrained_tvars

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
					"loss":loss, 
					"logits":masked_lm_log_probs,
					"masked_lm_example_loss":masked_lm_example_loss,
					"tvars":tvars,
					"model":model,
					"masked_lm_mask":masked_lm_mask,
					"output_ids":output_ids,
					"masked_lm_ids":masked_lm_ids
				}
		return return_dict
	return model_fn