import tensorflow as tf
import numpy as np

try:
	from .discriminator import model_fn_builder as discriminator
	from .generator import model_fn_builder as generator
	from .token_discriminator import discriminator_metric_train, discriminator_metric_eval
	from .token_generator import generator_metric_fn_train, generator_metric_fn_eval
except:
	from discriminator import model_fn_builder as discriminator
	from generator import model_fn_builder as generator
	from token_discriminator import discriminator_metric_train, discriminator_metric_eval

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
		discriminator_features['input_ids'] = generator_dict['sampled_ids']
		discriminator_features['input_mask'] = generator_dict['sampled_input_mask']
		discriminator_features['segment_ids'] = generator_dict['sampled_segment_ids']
		discriminator_features['input_ori_ids'] = generator_dict['sampled_input_ids']
		discriminator_features['next_sentence_labels'] = features['next_sentence_labels']
		discriminator_dict = discriminator_fn(discriminator_features, labels, mode, params)

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = []
		loss = discriminator_dict['loss']
		tvars.extend(discriminator_dict['tvars'])
		if kargs.get('joint_train', '0') == '1':
			tvars.extend(generator_fn['tvars'])
			loss += generator_dict['loss']

		var_checkpoint_dict_list = []
		for key in init_checkpoint_dict:
			if load_pretrained_dict[key] == "yes":
				if key == 'generator':
					tmp = {
						"tvars":generator_dict['tvars'],
						"init_checkpoint":init_checkpoint_dict['generator'],
						"exclude_scope":exclude_scope_dict[key]
					}
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

			if kargs.get('summary_debug', True):
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
								scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(
								mode=mode, 
								loss=loss, 
								train_op=train_op)

			return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:

			if kargs.get('joint_train', '0') == '1':
				generator_metric = generator_metric_fn_eval(
									generator_dict['masked_lm_example_loss'],
									generator_dict['masked_lm_log_probs'],
									generator_dict['masked_lm_ids'],
									generator_dict['masked_lm_weights'],
									generator_dict.get('next_sentence_example_loss', None),
									generator_dict.get('next_sentence_log_probs', None),
									generator_dict.get('next_sentence_labels', None)
									)
				eval_generator_metric = [(generator_metric_fn_eval, [
						  				generator_dict['masked_lm_example_loss'],
										generator_dict['masked_lm_log_probs'],
										generator_dict['masked_lm_ids'],
										generator_dict['masked_lm_weights'],
										generator_dict.get('next_sentence_example_loss', None),
										generator_dict.get('next_sentence_log_probs', None),
										generator_dict.get('next_sentence_labels', None)])]
			else:
				generator_metric = {}
				eval_generator_metric = []

			discriminator_metric = discriminator_metric_eval(
							discriminator_dict['per_example_loss'],
							discriminator_dict['logits'], 
							generator_dict['sampled_input_ids'], 
							generator_dict['sampled_ids'],
							generator_dict['sampled_input_mask'])
			eval_metrics = [(discriminator_metric_eval, [
						  				discriminator_dict['per_example_loss'],
										discriminator_dict['logits'], 
										generator_dict['sampled_input_ids'], 
										generator_dict['sampled_ids'],
										generator_dict['sampled_input_mask']
						])]

			metric_dict = discriminator_metric
			if len(generator_metric):
				metric_dict.update(discriminator_metric)

			if len(eval_generator_metric):
				eval_metrics.extend(eval_generator_metric)

			if kargs.get('use_tpu', False):
				estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
							  mode=mode,
							  loss=loss,
							  eval_metrics=eval_metrics,
							  scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=metric_dict)

			return estimator_spec
		else:
			raise NotImplementedError()

	return model_fn


