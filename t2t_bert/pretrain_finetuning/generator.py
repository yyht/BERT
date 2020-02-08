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

		if kargs.get('random_generator', '1') == '1':
			if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.TRAIN]:
				input_ori_ids = features['input_ori_ids']

				[output_ids, 
				sampled_binary_mask] = random_input_ids_generation(model_config,
											features['input_ori_ids'],
											features['input_mask'],
                                                                                        mask_probability=0.2,
                                                                                        replace_probability=0.1,
                                                                                        original_probability=0.1,
											**kargs)
				features['input_ids'] = output_ids
				tf.logging.info("****** do random generator *******")
			else:
				sampled_binary_mask = None
		else:
			sampled_binary_mask = None

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

		sampled_ids = token_generator(model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(), 
									features['input_ids'], 
									features['input_ori_ids'],
									features['input_mask'],	
									embedding_projection=model.get_embedding_projection_table(),
									scope=generator_scope_prefix,
									mask_method='only_mask',
									use_tpu=kargs.get('use_tpu', True))

		if model_config.get('gen_sample', 1) == 1:
			input_ids = features['input_ori_ids']
			input_mask = features['input_mask']
			segment_ids = features['segment_ids']
		else:
			input_ids = tf.expand_dims(features['input_ori_ids'], axis=-1)
			# batch x seq_length x 1
			input_ids = tf.einsum('abc,cd->abd', input_ids, tf.ones((1, model_config.get('gen_sample', 1))))
			input_ids = tf.cast(input_ids, tf.int32)

			input_shape_list = bert_utils.get_shape_list(input_ids, expected_rank=3)
			batch_size = input_shape_list[0]
			seq_length = input_shape_list[1]
			gen_sample = input_shape_list[2]

			sampled_ids = tf.reshape(sampled_ids, [batch*gen_sample, seq_length])
			input_ids = tf.reshape(input_ids, [batch*gen_sample, seq_length])

			input_mask = tf.expand_dims(features['input_mask'], axis=-1)
			input_mask = tf.einsum('abc,cd->abd', input_mask, tf.ones((1, model_config.get('gen_sample', 1))))
			input_mask = tf.cast(input_mask, tf.int32)

			segment_ids = tf.expand_dims(features['segmnet_ids'], axis=-1)
			segment_ids = tf.einsum('abc,cd->abd', segment_ids, tf.ones((1, model_config.get('gen_sample', 1))))
			segment_ids = tf.cast(segment_ids, tf.int32)

			segment_ids = tf.reshape(segment_ids, [batch*gen_sample, seq_length])
			input_mask = tf.reshape(input_mask, [batch*gen_sample, seq_length])

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
		tf.add_to_collection("generator_loss", loss)
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
					"sampled_binary_mask":sampled_binary_mask
				}
		return return_dict
	return model_fn
		
