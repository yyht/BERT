import tensorflow as tf
import numpy as np
from collections import Counter

try:
	from .cls_task import model_fn_builder as cls_model_fn
except:
	from cls_task import model_fn_builder as cls_model_fn

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

		total_loss = tf.constant(0.0)

		task_num = 0

		for index, task_type in enumerate(task_type_dict.keys()):
			if model_config_dict[task_type].model_type in model_type_lst:
				reuse = True
			else:
				reuse = None
				model_type_lst.append(model_config_dict[task_type].model_type)
			if task_type_dict[task_type] == "cls_task":
				model_fn = cls_model_fn(model_config_dict[task_type],
												num_labels_dict[task_type],
												init_checkpoint_dict[task_type],
												reuse,
												load_pretrained_dict[task_type],
												model_io_config,
												opt_config,
												exclude_scope=exclude_scope_dict[task_type],
												not_storage_params=[],
												target=target_dict[task_type],
												label_lst=None,
												output_type=output_type,
												task_layer_reuse=task_layer_reuse,
												task_type=task_type
												**kargs)
				result_dict = model_fn(features)
				logits_dict[key] = result_dict["logits"]
				losses_dict[key] = result_dict["loss"] # task loss
				total_loss += result_dict["loss"]
				task_num += result_dict["task_num"]
			else:
				continue

			if mode == tf.estimator.ModeKeys.TRAIN:
				train_ops.append(result_dict["train_op"])
				train_hooks.extend(result_dict["training_hooks"])
			elif mode == tf.estimator.ModeKeys.EVAL:
				features[key] = result_dict["feature"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			if output_type == "sess":
				return {
					"train":{
							"total_loss":total_loss/(1e-10+task_num), 
							"loss":losses
							"logits":logits,
							"train_op":tf.group(train_ops)
					},
					"hooks":train_hooks
				}
			elif output_type == "estimator":
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=total_loss/(1e-10+task_num),
								train_op=tf.group(train_ops),
								training_hooks=training_hooks)
				return estimator_spec

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
							"logits":logits,
							"total_loss":total_loss/(1e-10+task_num)
							"feature":features,
							"loss":losses
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

			