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

from task_module import classifier
from task_module import tsa_pretrain
import tensorflow as tf
from metric import tf_metrics
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
	print('==before scope==', model_config.scope)

	model_config = copy.deepcopy(model_config)
	model_config.num_train_steps = opt_config.num_train_steps
	model_config.scope = 'teacher/' + model_config.scope
	model_config.dropout_prob = 0.0
	model_config.attention_probs_dropout_prob = 0.0
	model_config.hidden_dropout_prob = 0.0
	generator_scope_prefix = 'teacher'

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)
		model = model_api(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)
		input_ori_ids = features['input_ori_ids']

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		sampled_binary_mask = features.get('masked_lm_mask', None)

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
										scope=generator_scope_prefix,
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
											scope=generator_scope_prefix,
											embedding_projection=model.get_embedding_projection_table())
		print(model_config.lm_ratio, '==mlm lm_ratio==')
		loss = model_config.lm_ratio * masked_lm_loss #+ 0.0 * nsp_loss
		
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

		pretrained_tvars.extend(lm_pretrain_tvars)
		tvars = pretrained_tvars

		print('==generator parameters==', tvars)

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(tvars, 
											init_checkpoint,
											exclude_scope=generator_scope_prefix,
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
					"masked_lm_ids":masked_lm_ids
				}
		return return_dict
	return model_fn