import tensorflow as tf
import numpy as np
import re

try:
	from .discriminator_gumbel import model_fn_builder as discriminator
	from .generator_gumbel import model_fn_builder as generator
	from .token_discriminator import discriminator_metric_train, discriminator_metric_eval
	from .token_generator import generator_metric_fn_train, generator_metric_fn_eval
except:
	from discriminator_gumbel import model_fn_builder as discriminator
	from generator_gumbel import model_fn_builder as generator
	from token_discriminator import discriminator_metric_train, discriminator_metric_eval
	from token_generator import generator_metric_fn_train, generator_metric_fn_eval

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

		graph = tf.Graph()
		with graph.as_default():

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
			if kargs.get('minmax_mode', 'corrupted') == 'corrupted':
				tf.logging.info("****** gumbel 3-D sampled_ids *******")
			elif kargs.get('minmax_mode', 'corrupted') == 'masked':
				discriminator_features['ori_sampled_ids'] = generator_dict['output_ids']
				tf.logging.info("****** conditioanl sampled_ids *******")
			discriminator_features['input_ids'] = generator_dict['sampled_ids']
			discriminator_features['input_mask'] = generator_dict['sampled_input_mask']
			discriminator_features['segment_ids'] = generator_dict['sampled_segment_ids']
			discriminator_features['input_ori_ids'] = generator_dict['sampled_input_ids']
			discriminator_features['next_sentence_labels'] = features['next_sentence_labels']
			discriminator_features['ori_input_ids'] = generator_dict['sampled_ids']
			
			discriminator_dict = discriminator_fn(discriminator_features, labels, mode, params)

			model_io_fn = model_io.ModelIO(model_io_config)

			tvars = []
			loss = kargs.get('dis_loss', 1.0) * discriminator_dict['loss']

			tvars.extend(discriminator_dict['tvars'])

			if kargs.get('joint_train', '1') == '1':
				tf.logging.info("****** joint generator and discriminator training *******")
				tvars.extend(generator_dict['tvars'])
				loss += generator_dict['loss']
			tvars = list(set(tvars))

			logging_hook = tf.train.LoggingTensorHook({"loss":loss, 
							"generator_loss" : tf.get_collection('generator_loss'),
							"discriminator_loss":tf.get_collection('discriminator_loss')},
							every_n_iter=1000)

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

			use_tpu = 1 if kargs.get('use_tpu', False) else 0
				
			if len(var_checkpoint_dict_list) >= 1:
				scaffold_fn = model_io_fn.load_multi_pretrained(var_checkpoint_dict_list,
												use_tpu=use_tpu)
			else:
				scaffold_fn = None

			if mode == tf.estimator.ModeKeys.TRAIN:

				if kargs.get('summary_debug', False):
					metric_dict = discriminator_metric_train(discriminator_dict['per_example_loss'],
									discriminator_dict['logits'], 
								generator_dict['sampled_input_ids'], 
								generator_dict['sampled_ids'],
								generator_dict['sampled_input_mask'])

					for key in metric_dict:
						tf.summary.scalar(key, metric_dict[key])
		
				if kargs.get('use_tpu', False):
					optimizer_fn = optimizer.Optimizer(opt_config)
					use_tpu = 1
				else:
					optimizer_fn = distributed_optimizer.Optimizer(opt_config)
					use_tpu = 0

				model_io_fn.print_params(tvars, string=", trainable params")
				
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(update_ops):
					train_op = optimizer_fn.get_train_op(loss, list(set(tvars)),
									opt_config.init_lr, 
									opt_config.num_train_steps,
									use_tpu=use_tpu)

				if kargs.get('use_tpu', False):
					estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
									mode=mode,
									loss=loss,
									train_op=train_op,
									scaffold_fn=scaffold_fn,
									training_hooks=[logging_hook]
									)
				else:
					estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode, 
									loss=loss, 
									train_op=train_op)

				return estimator_spec

			elif mode == tf.estimator.ModeKeys.EVAL:

				if kargs.get('joint_train', '0') == '1':

					def joint_metric(masked_lm_example_loss, masked_lm_log_probs,
									masked_lm_ids, masked_lm_weights,
									next_sentence_example_loss, next_sentence_log_probs,
									next_sentence_labels,
									per_example_loss, logits,
									input_ori_ids, input_ids,
									input_mask):
						generator_metric = generator_metric_fn_eval(
											masked_lm_example_loss,
											masked_lm_log_probs,
											masked_lm_ids,
											masked_lm_weights,
											next_sentence_example_loss,
											next_sentence_log_probs,
											next_sentence_labels
											)
						discriminator_metric = discriminator_metric_eval(
								per_example_loss,
								logits, 
								input_ori_ids, 
								input_ids,
								input_mask)
						generator_metric.update(discriminator_metric)
						return generator_metric

					tpu_eval_metrics = (joint_metric, [
											generator_dict['masked_lm_example_loss'],
											generator_dict['masked_lm_log_probs'],
											generator_dict['masked_lm_ids'],
											generator_dict['masked_lm_weights'],
											generator_dict.get('next_sentence_example_loss', None),
											generator_dict.get('next_sentence_log_probs', None),
											generator_dict.get('next_sentence_labels', None),
											discriminator_dict['per_example_loss'],
											discriminator_dict['logits'], 
											generator_dict['sampled_input_ids'], 
											generator_dict['sampled_ids'],
											generator_dict['sampled_input_mask']])
					gpu_eval_metrics = joint_metric(generator_dict['masked_lm_example_loss'],
											generator_dict['masked_lm_log_probs'],
											generator_dict['masked_lm_ids'],
											generator_dict['masked_lm_weights'],
											generator_dict.get('next_sentence_example_loss', None),
											generator_dict.get('next_sentence_log_probs', None),
											generator_dict.get('next_sentence_labels', None),
											discriminator_dict['per_example_loss'],
											discriminator_dict['logits'], 
											generator_dict['sampled_input_ids'], 
											generator_dict['sampled_ids'],
											generator_dict['sampled_input_mask'])
				else:
					gpu_eval_metrics = discriminator_metric_eval(
									discriminator_dict['per_example_loss'],
									discriminator_dict['logits'], 
									generator_dict['sampled_input_ids'], 
									generator_dict['sampled_ids'],
									generator_dict['sampled_input_mask'])
					tpu_eval_metrics = (discriminator_metric_eval, [
												discriminator_dict['per_example_loss'],
												discriminator_dict['logits'], 
												generator_dict['sampled_input_ids'], 
												generator_dict['sampled_ids'],
												generator_dict['sampled_input_mask']
								])		

				if kargs.get('use_tpu', False):
					estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
								  mode=mode,
								  loss=loss,
								  eval_metrics=tpu_eval_metrics,
								  scaffold_fn=scaffold_fn)
				else:
					estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
									loss=loss,
									eval_metric_ops=gpu_eval_metrics)

				return estimator_spec
			else:
				raise NotImplementedError()

	return model_fn


