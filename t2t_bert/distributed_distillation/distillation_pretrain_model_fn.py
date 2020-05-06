import tensorflow as tf
import numpy as np

try:
	from .student_model_pretrain_fn import model_fn_builder as st_model_fn
	from .teacher_model_pretrain_fn import model_fn_builder as ta_model_fn
except:
	from student_model_pretrain_fn import model_fn_builder as st_model_fn
	from teacher_model_pretrain_fn import model_fn_builder as ta_model_fn

from model_io import model_io
from optimizer import distributed_optimizer as optimizer
from distillation import distillation_utils 
from distillation import flip_gradient
from distillation import mdd_utils
from distillation import repo_distillation_utils
from metric import tf_metrics
from distillation import uniform_mapping
from distillation import cpc_utils
from utils.bert import bert_utils

def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
					masked_lm_weights, name):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
	masked_lm_accuracy = tf.metrics.accuracy(
		labels=masked_lm_ids,
		predictions=masked_lm_predictions,
		weights=masked_lm_weights)
	masked_lm_mean_loss = tf.metrics.mean(
		values=masked_lm_example_loss, weights=masked_lm_weights)

	return {
		name+"/masked_lm_accuracy": masked_lm_accuracy,
		name+"/masked_lm_loss": masked_lm_mean_loss
		}

def train_metric_fn(masked_lm_example_loss, masked_lm_log_probs, 
					masked_lm_ids,
					masked_lm_weights, 
					name,
					**kargs):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
	masked_lm_weights = tf.cast(masked_lm_weights, tf.float32)

	masked_lm_mask = kargs.get('masked_lm_mask', None)
	if masked_lm_mask is not None:
		masked_lm_mask = tf.reshape(masked_lm_mask, [-1])
		masked_lm_weights *= tf.cast(masked_lm_mask, tf.float32)

	masked_lm_accuracy = tf.equal(
						tf.cast(masked_lm_ids, tf.int32),
						tf.cast(masked_lm_predictions, tf.int32)
					)
	masked_lm_accuracy = tf.cast(masked_lm_accuracy, tf.int32)*tf.cast(masked_lm_weights, dtype=tf.int32)
	masked_lm_accuracy = tf.reduce_sum(tf.cast(masked_lm_accuracy, tf.float32)) / tf.reduce_sum(masked_lm_weights)
	masked_lm_mean_loss = tf.reduce_sum(masked_lm_example_loss*masked_lm_weights) / tf.reduce_sum(masked_lm_weights)

	return {
		name+"/masked_lm_accuracy": masked_lm_accuracy,
		name+"/masked_lm_loss": masked_lm_mean_loss,
		name+"/valid_position":tf.reduce_sum(masked_lm_weights)
		}

def distillation_model_fn(model_config_dict,
					num_labels_dict,
					init_checkpoint_dict,
					load_pretrained_dict,
					model_io_config={},
					opt_config={},
					exclude_scope_dict={},
					not_storage_params_dict={},
					target_dict={},
					output_type="sess",
					distillation_config={},
					**kargs):

	def model_fn(features, labels, mode, params):

		original_loss = tf.constant(0.0)
		distilled_loss = tf.constant(0.0)

		st_model = st_model_fn(model_config_dict['student'],
		 			num_labels_dict['student'],
					init_checkpoint_dict['student'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['student'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('student', ""),
					not_storage_params=not_storage_params_dict.get('student', []),
					target=target_dict['student'],
					**kargs)
		st_dict = st_model(features, labels, mode, params)

		# ta_model = ta_model_fn(model_config_dict['teacher'],
		#  			num_labels_dict['teacher'],
		# 			init_checkpoint_dict['teacher'],
		# 			model_reuse=None,
		# 			load_pretrained=load_pretrained_dict['teacher'],
		# 			model_io_config=model_io_config,
		# 			opt_config=opt_config,
		# 			exclude_scope=exclude_scope_dict.get('teacher', ""),
		# 			not_storage_params=not_storage_params_dict.get('teacher', []),
		# 			target=target_dict['teacher'],
		# 			**kargs)
		# ta_features = {}
		# for key in features:
		# 	ta_features[key] = features[key]
		# ta_features['masked_lm_mask'] = st_dict['masked_lm_mask']
		# ta_features['input_ids'] = st_dict['output_ids']
		# ta_features['input_ori_ids'] = features['input_ids']
		# ta_dict = ta_model(ta_features, labels, mode, params)

		# studnet_logit = st_dict['logits']
		# teacher_logit = ta_dict['logits']

		model_io_fn = model_io.ModelIO(model_io_config)

		original_loss += st_dict['loss'] * (distillation_config.get('ce_loss', 1.0))
		print(distillation_config.get('ce_loss', 1.0), '===ce_loss===')
		if not kargs.get('use_tpu', False):
			tf.summary.scalar("ce_loss", st_dict['loss'])

		hook_dict = {}

		# if 'kl_logits' in distillation_config.get('distillation_type', ['kl_logits']):
		# 	temperature = distillation_config.get('kl_temperature', 2.0)
		# 	distilled_teacher_logit = tf.nn.log_softmax((teacher_logit+1e-10) / temperature) # log_softmax logits
		# 	distilled_student_logit = tf.nn.log_softmax((studnet_logit+1e-10) / temperature) # log_softmax logits

		# 	logits_mask = tf.cast(st_dict['masked_lm_mask'], tf.float32)
		# 	kl_distilled_loss = distillation_utils.kd(distilled_teacher_logit, 
		# 												distilled_student_logit)
		# 	kl_distilled_loss = tf.reduce_sum(logits_mask*kl_distilled_loss) / tf.reduce_sum(logits_mask)

		# 	if not kargs.get('use_tpu', False):
		# 		tf.summary.scalar("kl_logits_loss", kl_distilled_loss)
		# 		tf.summary.scalar("kl_logits_mask", tf.reduce_mean(logits_mask))
		# 	tf.logging.info("***** with knowledge distillation %s tenperature *****", str(temperature))
		# 	hook_dict['kl_logits_loss'] = kl_distilled_loss
		# 	# kl_distilled_loss *= np.power(temperature, 2)
		# 	distilled_loss += kl_distilled_loss * distillation_config.get('kl_logits', 0.9)
		# 	print(distillation_config.get('kl_logits_ratio', 0.9), '===kl_logits_ratio===')

		# if "attention_score_uniform" in distillation_config.get('distillation_type', ['kl_logits']):
		# 	source_attention_score = ta_dict['model'].get_multihead_attention()
		# 	target_attention_score = st_dict['model'].get_multihead_attention()

		# 	print("==apply attention_score_uniform==")

		# 	with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):  
		# 		attention_loss = uniform_mapping.attention_score_matching(source_attention_score, 
		# 																target_attention_score,
		# 																features['input_mask'],
		# 																0)
		# 	tf.summary.scalar("attention_score_uniform_loss", attention_loss)
		# 	distilled_loss += attention_loss * distillation_config.get("attention_score_uniform", 0.1)
		# 	hook_dict['attention_mse_loss'] = attention_loss
		# 	print(distillation_config.get('attention_score_uniform', 0.1), '===attention_score_uniform===')
			
		# if "hidden_uniform" in distillation_config.get('distillation_type', ['kl_logits']):
		# 	source_hidden = ta_dict['model'].get_all_encoder_layers()
		# 	target_hidden = st_dict['model'].get_all_encoder_layers()

		# 	print("==apply hidden_uniform==")

		# 	with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):
		# 		hidden_loss = uniform_mapping.hidden_matching(source_hidden, target_hidden, 
		# 													features['input_mask'],
		# 													0)
		# 	if not kargs.get('use_tpu', False):
		# 		tf.summary.scalar("hidden_uniform_loss", hidden_loss)
		# 	distilled_loss += hidden_loss * distillation_config.get("hidden_uniform", 0.1)
		# 	hook_dict['hidden_loss'] = hidden_loss
		# 	print(distillation_config.get('hidden_uniform', 0.1), '===hidden_uniform===')

		# if "embedding_distillation" in distillation_config.get('distillation_type', ['embedding_distillation']):
		# 	st_word_embed = st_dict['model'].get_embedding_table()
		# 	ta_word_embed = ta_dict['model'].get_embedding_table()
		# 	st_word_embed_shape = bert_utils.get_shape_list(st_word_embed, expected_rank=[2,3])
		# 	print("==random_embed_shape==", st_word_embed_shape)
		# 	ta_word_embed_shape = bert_utils.get_shape_list(ta_word_embed, expected_rank=[2,3])
		# 	print("==pretrain_embed_shape==", ta_word_embed_shape)
		# 	if st_word_embed_shape[-1] != ta_word_embed_shape[-1]:
		# 		with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):
		# 			with tf.variable_scope("embedding_proj"):
		# 				proj_embed = tf.layers.dense(ta_word_embed, st_word_embed_shape[-1])
		# 	else:
		# 		proj_embed = ta_word_embed
			
		# 	embed_loss = tf.reduce_mean(tf.reduce_mean(tf.square(proj_embed-st_word_embed), axis=-1))
		# 	distilled_loss += embed_loss
		# 	hook_dict['embed_loss'] = embed_loss
		# 	tf.logging.info("****** apply prertained feature distillation *******")

		total_loss = distilled_loss + original_loss
		tvars = []
		tvars.extend(st_dict['tvars'])

		distillation_vars = model_io_fn.get_params('distillation', 
							not_storage_params=[])
		tvars.extend(distillation_vars)

		# if kargs.get('update_ta', False):
		# 	total_loss += ta_dict['loss']
		# 	tvars.extend(ta_dict['tvars'])

		if not kargs.get('use_tpu', False):
			student_eval_metrics = train_metric_fn(
						  st_dict['masked_lm_example_loss'], 
						  st_dict['logits'], 
						  st_dict["masked_lm_ids"],
						  st_dict['masked_lm_mask'],
						  'student')

			# teacher_eval_metric =  train_metric_fn( 
			# 			  ta_dict['masked_lm_example_loss'], 
			# 			  ta_dict['logits'], 
			# 			  ta_dict["masked_lm_ids"],
			# 			  ta_dict['masked_lm_mask'],
			# 			  'teacher')
			# student_eval_metrics.update(teacher_eval_metric)
			for key in student_eval_metrics:
				hook_dict[key] = student_eval_metrics[key]

		if mode == tf.estimator.ModeKeys.TRAIN:

			optimizer_fn = optimizer.Optimizer(opt_config)

			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print("==update_ops==", update_ops)

			print('==total trainable vars==', list(tvars))

			with tf.control_dependencies(update_ops):
				train_op = optimizer_fn.get_train_op(total_loss, list(set(tvars)), 
								opt_config.init_lr, 
								opt_config.num_train_steps,
								**kargs)

				model_io_fn.set_saver()

				if kargs.get("task_index", 1) == 1 and kargs.get("run_config", None):
					training_hooks = []
				elif kargs.get("task_index", 1) == 1:
					model_io_fn.get_hooks(kargs.get("checkpoint_dir", None), 
														kargs.get("num_storage_steps", 1000))

					training_hooks = model_io_fn.checkpoint_hook
				else:
					training_hooks = []

				logging_hook = tf.train.LoggingTensorHook(
					hook_dict, every_n_iter=100)
				training_hooks.append(logging_hook)

				if len(optimizer_fn.distributed_hooks) >= 1:
					training_hooks.extend(optimizer_fn.distributed_hooks)
				print(training_hooks, "==training_hooks==", "==task_index==", kargs.get("task_index", 1))

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=total_loss, train_op=train_op,
								training_hooks=training_hooks)
				
				return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:
		
			student_eval_metrics = metric_fn(
								  st_dict['masked_lm_example_loss'], 
								  st_dict['logits'], 
								  st_dict["masked_lm_ids"],
								  st_dict['masked_lm_mask'],
								  'student')

			# teacher_eval_metric =  metric_fn( 
			# 					  ta_dict['masked_lm_example_loss'], 
			# 					  ta_dict['logits'], 
			# 					  ta_dict["masked_lm_ids"],
			# 					  ta_dict['masked_lm_mask'],
			# 					  'teacher')

			# student_eval_metrics.update(teacher_eval_metric)

			estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
							loss=total_loss,
							eval_metric_ops=student_eval_metrics)
			return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

