import tensorflow as tf
import numpy as np

try:
	from .student_model_fn import model_fn_builder as st_model_fn
	from .teacher_model_fn import model_fn_builder as ta_model_fn
except:
	from student_model_fn import model_fn_builder as st_model_fn
	from teacher_model_fn import model_fn_builder as ta_model_fn

from model_io import model_io
from optimizer import distributed_optimizer as optimizer
from distillation import distillation_utils 
from distillation import flip_gradient
from distillation import mdd_utils
from distillation import repo_distillation_utils
from metric import tf_metrics
from distillation import uniform_mapping

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

	def model_fn(features, labels, mode):

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
		st_dict = st_model(features, labels, mode)

		ta_model = ta_model_fn(model_config_dict['teacher'],
		 			num_labels_dict['teacher'],
					init_checkpoint_dict['teacher'],
					model_reuse=None,
					load_pretrained=load_pretrained_dict['teacher'],
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope=exclude_scope_dict.get('teacher', ""),
					not_storage_params=not_storage_params_dict.get('teacher', []),
					target=target_dict['teacher'],
					**kargs)
		ta_dict = ta_model(features, labels, mode)

		studnet_logit = st_dict['logits']
		teacher_logit = ta_dict['logits']

		model_io_fn = model_io.ModelIO(model_io_config)

		feature_flag = False

		original_loss += st_dict['loss'] * (distillation_config.get('ce_loss', 1.0))
		print(distillation_config.get('ce_loss', 1.0), '===ce_loss===')
		tf.summary.scalar("ce_loss", st_dict['loss'])

		if 'kl_logits' in distillation_config.get('distillation_type', ['kl_logits']):
			temperature = distillation_config.get('kl_temperature', 2.0)
			distilled_teacher_logit = tf.nn.log_softmax((teacher_logit+1e-10) / temperature) # log_softmax logits
			distilled_student_logit = tf.nn.log_softmax((studnet_logit+1e-10) / temperature) # log_softmax logits

			kl_distilled_loss = tf.reduce_mean(distillation_utils.kd(distilled_teacher_logit, 
														distilled_student_logit))

			tf.summary.scalar("kl_logits_loss", kl_distilled_loss)
			
			# kl_distilled_loss *= np.power(temperature, 2)
			distilled_loss += kl_distilled_loss * distillation_config.get('kl_logits_ratio', 0.9)
			print(distillation_config.get('kl_logits_ratio', 0.9), '===kl_logits_ratio===')

		if 'rkd' in distillation_config.get('distillation_type', ['kl_logits']):
			source = ta_dict['model'].get_pooled_output()
			target = st_dict['model'].get_pooled_output()
			print("==apply rkd==")
			with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):  
				rkd_loss = repo_distillation_utils.RKD(source, target, l = [25,50])
			tf.summary.scalar("rkd_loss", rkd_loss)
			distilled_loss += rkd_loss * distillation_config.get("rkd_ratio", 0.1)

		if "attention_score_uniform" in distillation_config.get('distillation_type', ['kl_logits']):
			source_attention_score = ta_dict['model'].get_multihead_attention()
			target_attention_score = st_dict['model'].get_multihead_attention()

			print("==apply attention_score_uniform==")

			with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):  
				attention_loss = uniform_mapping.attention_score_matching(source_attention_score, 
																		target_attention_score,
																		0)
			tf.summary.scalar("attention_score_uniform_loss", attention_loss)
			feature_flag = True
			distilled_loss += attention_loss * distillation_config.get("attention_score_uniform", 0.1)

			print(distillation_config.get('attention_score_uniform', 0.1), '===attention_score_uniform===')
			
		if "hidden_uniform" in distillation_config.get('distillation_type', ['kl_logits']):
			source_hidden = ta_dict['model'].get_all_encoder_layers()
			target_hidden = st_dict['model'].get_all_encoder_layers()

			print("==apply hidden_uniform==")

			with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):
				hidden_loss = uniform_mapping.hidden_matching(source_hidden, target_hidden, 0)
			tf.summary.scalar("hidden_uniform_loss", hidden_loss)
			distilled_loss += hidden_loss * distillation_config.get("hidden_uniform", 0.1)
			feature_flag = True

			print(distillation_config.get('hidden_uniform', 0.1), '===hidden_uniform===')

		if "hidden_cls_uniform" in distillation_config.get('distillation_type', ['kl_logits']):
			source_hidden = ta_dict['model'].get_all_encoder_layers()
			target_hidden = st_dict['model'].get_all_encoder_layers()

			print("==apply hidden_cls_uniform==")
			with tf.variable_scope("distillation", reuse=tf.AUTO_REUSE):
				hidden_cls_loss = uniform_mapping.hidden_cls_matching(source_hidden, target_hidden, 0)
			tf.summary.scalar("hidden_cls_uniform_loss", hidden_cls_loss)
			distilled_loss += hidden_cls_loss * distillation_config.get("hidden_uniform", 0.1)
			feature_flag = True


		if "mdd" in distillation_config.get('distillation_type', ['mdd']):
			source = ta_dict['model'].get_pooled_output()
			target = st_dict['model'].get_pooled_output()

			print("==apply mdd==")


		total_loss = distilled_loss + original_loss

		tvars = []
		tvars.extend(st_dict['tvars'])

		if feature_flag:
			distillation_vars = model_io_fn.get_params('distillation', 
								not_storage_params=[])
			tvars.extend(distillation_vars)

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

				if kargs.get("task_index", 1) == 0 and kargs.get("run_config", None):
					training_hooks = []
				elif kargs.get("task_index", 1) == 0:
					model_io_fn.get_hooks(kargs.get("checkpoint_dir", None), 
														kargs.get("num_storage_steps", 1000))

					training_hooks = model_io_fn.checkpoint_hook
				else:
					training_hooks = []

				if len(optimizer_fn.distributed_hooks) >= 1:
					training_hooks.extend(optimizer_fn.distributed_hooks)
				print(training_hooks, "==training_hooks==", "==task_index==", kargs.get("task_index", 1))

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=total_loss, train_op=train_op,
								training_hooks=training_hooks)
				if output_type == "sess":
					return {
						"train":{
										"loss":total_loss, 
										"logits":studnet_logit,
										"train_op":train_op
								},
						"hooks":training_hooks
					}
				elif output_type == "estimator":
					return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(per_example_loss,
						logits, 
						label_ids, model_type):
				"""Computes the loss and accuracy of the model."""
				sentence_log_probs = tf.reshape(
					logits, [-1, logits.shape[-1]])
				sentence_predictions = tf.argmax(
					logits, axis=-1, output_type=tf.int32)
				sentence_labels = tf.reshape(label_ids, [-1])
				sentence_accuracy = tf.metrics.accuracy(
					labels=label_ids, predictions=sentence_predictions)
				sentence_mean_loss = tf.metrics.mean(
					values=per_example_loss)
				sentence_f = tf_metrics.f1(label_ids, 
										sentence_predictions, 
										num_labels_dict['student'], 
										None, average="macro")

				eval_metric_ops = {
									"{}_f1".format(model_type): sentence_f,
									"{}_acc".format(model_type):sentence_accuracy
								}

				return eval_metric_ops

			if output_type == "sess":
				return {
					"eval":{
							"per_example_loss":st_dict['logits']['per_example_loss'],
							"logits":studnet_logit,
							"loss":tf.reduce_mean(st_dict['logits']['per_example_loss']),
							"feature":st_dict['model'].get_pooled_output()
						}
				}
			elif output_type == "estimator":
				eval_metric_ops = metric_fn( 
							st_dict['per_example_loss'],
							studnet_logit, 
							features['label_ids'],
							"student")
				teacher_eval_metric_ops =  metric_fn( 
							ta_dict['per_example_loss'],
							teacher_logit, 
							features['label_ids'],
							"teacher")

				eval_metric_ops.update(teacher_eval_metric_ops)

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=total_loss,
								eval_metric_ops=eval_metric_ops)
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

