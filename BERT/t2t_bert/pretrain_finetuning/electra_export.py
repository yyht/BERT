import tensorflow as tf
import numpy as np
import re

try:
	from .discriminator_exporter import model_fn_builder as discriminator
	from .generator_exported import model_fn_builder as generator
except:
	from discriminator_exporter import model_fn_builder as discriminator
	from generator_exported import model_fn_builder as generator

import tensorflow as tf
import numpy as np
from optimizer import optimizer
from optimizer import distributed_optimizer

from model_io import model_io

import tensorflow as tf
from metric import tf_metrics

def classifier_model_fn_builder(
						model_config_dict,
						num_labels_dict,
						init_checkpoint_dict,
						load_pretrained_dict,
						model_io_config={},
						opt_config={},
						exclude_scope_dict={},
						not_storage_params_dict={},
						target_dict={},
						**kargs):
	
	def model_fn(features, labels, mode, params):

		generator_fn = generator(model_config_dict['generator'],
					num_labels_dict['generator'],
					init_checkpoint_dict['generator'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['generator'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('generator', ""),
					not_storage_params=not_storage_params_dict.get('generator', []),
					target=target_dict['generator'],
					**kargs)
		generator_dict = generator_fn(features, labels, mode, params)

		discriminator_fn = discriminator(model_config_dict['discriminator'],
					num_labels_dict['discriminator'],
					init_checkpoint_dict['discriminator'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['discriminator'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('discriminator', ""),
					not_storage_params=not_storage_params_dict.get('discriminator', []),
					target=target_dict['discriminator'],
					**kargs)

		discriminator_features = {}
		discriminator_features['input_ids'] = features['input_ids']
		discriminator_features['input_mask'] = features['input_mask']
		discriminator_features['segment_ids'] = features['segment_ids']
		discriminator_features['input_ori_ids'] = features['input_ori_ids']
		discriminator_features['next_sentence_labels'] = features['next_sentence_labels']
		discriminator_dict = discriminator_fn(discriminator_features, labels, mode, params)

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = []
		loss = discriminator_dict['loss']

		tvars.extend(discriminator_dict['tvars'])

		tvars.extend(generator_dict['tvars'])
		loss += generator_dict['loss']
		tvars = list(set(tvars))

		var_checkpoint_dict_list = []
		for key in init_checkpoint_dict:
			if load_pretrained_dict[key] == "yes":
				if key == 'generator':
					tmp = {
						"tvars":generator_dict['tvars'],
						"init_checkpoint":init_checkpoint_dict['generator'],
						"exclude_scope":exclude_scope_dict[key]
					}
					if kargs.get("sharing_mode", "none") != "none":
						tmp['exclude_scope'] = ''
					var_checkpoint_dict_list.append(tmp)
				elif key == 'discriminator':
					tmp = {
						"tvars":discriminator_dict['tvars'],
						"init_checkpoint":init_checkpoint_dict['discriminator'],
						"exclude_scope":exclude_scope_dict[key]
					}
					var_checkpoint_dict_list.append(tmp)
			
		if len(var_checkpoint_dict_list) >= 1:
			scaffold_fn = model_io_fn.load_multi_pretrained(var_checkpoint_dict_list,
											use_tpu=use_tpu)
		else:
			scaffold_fn = None

		if mode == tf.estimator.ModeKeys.PREDICT:

			masked_lm_mask = generator_dict['masked_lm_mask']
			masked_lm_log_probs = generator_dict['masked_lm_log_probs']

			dis_log_probs = discriminator_dict['logits']
			
			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'masked_lm_mask':masked_lm_mask,
												"masked_lm_log_probs":masked_lm_log_probs,
												"dis_log_probs":dis_log_probs
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'masked_lm_mask':masked_lm_mask,
														"masked_lm_log_probs":masked_lm_log_probs,
														"dis_log_probs":dis_log_probs
													}
												)
									}
						)
			return estimator_spec
	return model_fn

