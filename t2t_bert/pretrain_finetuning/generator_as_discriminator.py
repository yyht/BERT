import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

from pretrain_finetuning.token_generator import token_generator, random_input_ids_generation

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

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)
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
										reuse=tf.AUTO_REUSE)

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
			seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
			print("==apply bert masked lm==")

		(masked_lm_loss,
		masked_lm_example_loss, 
		masked_lm_log_probs,
		masked_lm_mask) = seq_masked_lm_fn(model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(),
									features['input_mask'], 
									features['input_ori_ids'], 
									features['input_ids'],
									features['input_mask'],
									reuse=tf.AUTO_REUSE,
									embedding_projection=model.get_embedding_projection_table())
		masked_lm_ids = features['input_ori_ids']
		
		loss = model_config.lm_ratio * masked_lm_loss + 0.0 * nsp_loss

		model_io_fn = model_io.ModelIO(model_io_config)
		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

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
		tf.add_to_collection("discriminator_loss", loss)
		return_dict = {
					"loss":loss, 
					"tvars":tvars,
					"model":model,
					"per_example_loss":masked_lm_example_loss,
					"masked_lm_weights":masked_lm_mask,
					"masked_lm_log_probs":masked_lm_log_probs,
					"next_sentence_example_loss":nsp_per_example_loss,
					"next_sentence_log_probs":nsp_log_prob, 
					"next_sentence_labels":features['next_sentence_labels']
				}
		return return_dict
	return model_fn
		
