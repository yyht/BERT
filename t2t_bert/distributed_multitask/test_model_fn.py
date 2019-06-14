import tensorflow as tf
import numpy as np
from collections import Counter

try:
	from .cls_task import model_fn_builder as cls_model_fn
except:
	from cls_task import model_fn_builder as cls_model_fn

from model_io import model_io
from optimizer import distributed_optimizer as optimizer

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

def multitask_model_fn(model_config_dict,
					num_labels_dict,
					task_type_dict,
					init_checkpoint_dict,
					load_pretrained_dict,
					model_io_config={},
					opt_config={},
					exclude_scope_dict={},
					not_storage_params_dict={},
					target_dict={},
					label_lst=None,
					output_type="sess",
					task_layer_reuse=None,
					model_type_lst=[],
					**kargs):

	def model_fn(features, labels, mode):

		train_ops = []
		train_hooks = []
		logits_dict = {}
		losses_dict = {}
		features_dict = {}
		tvars = []
		task_num_dict = {}

		total_loss = tf.constant(0.0)

		task_num = 0

		encoder = {}
		hook_dict = {}

		# print(task_type_dict.keys(), "==task type dict==")
		num_task = len(task_type_dict)

		for index, task_type in enumerate(task_type_dict.keys()):
			if model_config_dict[task_type].model_type in model_type_lst:
				reuse = True
			else:
				reuse = None
				model_type_lst.append(model_config_dict[task_type].model_type)
			if task_type_dict[task_type] == "cls_task":

				if model_config_dict[task_type].model_type not in encoder:
					model_api = model_zoo(model_config_dict[task_type])

					model = model_api(model_config_dict[task_type], features, labels,
							mode, target_dict[task_type], reuse=reuse)
					encoder[model_config_dict[task_type].model_type] = model

				print(encoder, "==encode==")

				task_model_fn = cls_model_fn(encoder[model_config_dict[task_type].model_type],
												model_config_dict[task_type],
												num_labels_dict[task_type],
												init_checkpoint_dict[task_type],
												reuse,
												load_pretrained_dict[task_type],
												model_io_config,
												opt_config,
												exclude_scope=exclude_scope_dict[task_type],
												not_storage_params=not_storage_params_dict[task_type],
												target=target_dict[task_type],
												label_lst=None,
												output_type=output_type,
												task_layer_reuse=task_layer_reuse,
												task_type=task_type,
												num_task=num_task,
												task_adversarial=1e-2,
												**kargs)
				print("==SUCCEEDED IN LODING==", task_type)

				result_dict = task_model_fn(features, labels, mode)
				# logits_dict[task_type] = result_dict["logits"]
				# losses_dict[task_type] = result_dict["loss"] # task loss
				# for key in ["masked_lm_loss", "task_loss"]:
				# 	name = "{}_{}".format(task_type, key)
				# 	if name in result_dict:
				# 		hook_dict[name] = result_dict[name]
				total_loss += result_dict["loss"]
				if mode == tf.estimator.ModeKeys.TRAIN:
					tvars.extend(result_dict["tvars"])
					task_num += result_dict["task_num"]
					task_num_dict[task_type] = result_dict["task_num"]
				elif mode == tf.estimator.ModeKeys.EVAL:
					features[task_type] = result_dict["feature"]
			else:
				continue

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn = model_io.ModelIO(model_io_config)

			optimizer_fn = optimizer.Optimizer(opt_config)

			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):

				train_op = optimizer_fn.get_train_op(total_loss, tvars, 
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

				return estimator_spec

		# if mode == tf.estimator.ModeKeys.TRAIN:
		# 	model_io_fn = model_io.ModelIO(model_io_config)

		# 	optimizer_fn = optimizer.Optimizer(opt_config)

		# 	model_io_fn.print_params(list(set(tvars)), string=", trainable params")
		# 	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# 	print("==update_ops==", update_ops)

		# 	with tf.control_dependencies(update_ops):
		# 		train_op = optimizer_fn.get_train_op(total_loss, list(set(tvars)), 
		# 						opt_config.init_lr, 
		# 						opt_config.num_train_steps,
		# 						**kargs)

		# 		model_io_fn.set_saver()

		# 		if kargs.get("task_index", 1) == 0 and kargs.get("run_config", None):
		# 			training_hooks = []
		# 		elif kargs.get("task_index", 1) == 0:
		# 			model_io_fn.get_hooks(kargs.get("checkpoint_dir", None), 
		# 												kargs.get("num_storage_steps", 1000))

		# 			training_hooks = model_io_fn.checkpoint_hook
		# 		else:
		# 			training_hooks = []

		# 		if len(optimizer_fn.distributed_hooks) >= 1:
		# 			training_hooks.extend(optimizer_fn.distributed_hooks)
		# 		print(training_hooks, "==training_hooks==", "==task_index==", kargs.get("task_index", 1))

		# 	if output_type == "sess":
		# 		return {
		# 			"train":{
		# 					"total_loss":total_loss, 
		# 					"loss":losses_dict,
		# 					"logits":logits_dict,
		# 					"train_op":train_op,
		# 					"task_num_dict":task_num_dict
		# 			},
		# 			"hooks":train_hooks
		# 		}
		# 	elif output_type == "estimator":

		# 		# hook_dict['learning_rate'] = optimizer_fn.learning_rate
		# 		# logging_hook = tf.train.LoggingTensorHook(
		# 		# 	hook_dict, every_n_iter=100)
		# 		# training_hooks.append(logging_hook)

		# 		estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
		# 						loss=total_loss,
		# 						train_op=train_op,
		# 						training_hooks=training_hooks)
		# 		return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL: # eval execute for each class solo
			def metric_fn(logits, 
						label_ids):
				"""Computes the loss and accuracy of the model."""
				sentence_log_probs = tf.reshape(
					logits, [-1, logits.shape[-1]])
				sentence_predictions = tf.argmax(
					logits, axis=-1, output_type=tf.int32)
				sentence_labels = tf.reshape(label_ids, [-1])
				sentence_accuracy = tf.metrics.accuracy(
					labels=label_ids, predictions=sentence_predictions)
				sentence_f = tf_metrics.f1(label_ids, 
										sentence_predictions, 
										num_labels, 
										label_lst, average="macro")

				eval_metric_ops = {
									"f1": sentence_f,
									"acc":sentence_accuracy
								}

				return eval_metric_ops

			if output_type == "sess":
				return {
					"eval":{
							"logits":logits_dict,
							"total_loss":total_loss,
							"feature":features,
							"loss":losses_dict
						}
				}
			elif output_type == "estimator":
				eval_metric_ops = {}
				for key in logits:
					eval_dict = metric_fn(
							logits[key],
							features_task_dict[key]["label_ids"]
						)
					for sub_key in eval_dict.keys():
						eval_key = "{}_{}".format(key, sub_key)
						eval_metric_ops[eval_key] = eval_dict[sub_key]
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=total_loss/task_num,
								eval_metric_ops=eval_metric_ops)
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

			