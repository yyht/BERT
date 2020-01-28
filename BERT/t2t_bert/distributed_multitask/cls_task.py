try:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils
except:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics
from task_module import pretrain

from optimizer import distributed_optimizer as optimizer

def build_accuracy(logits, labels, mask):
	pred_label = tf.argmax(logits, axis=-1)
	correct = tf.equal(
		tf.cast(pred_label, tf.int32),
		tf.cast(labels, tf.int32)
	)
	mask = tf.cast(mask, tf.float32)
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)*mask)/(1e-10+tf.reduce_sum(mask))
	return accuracy

def model_fn_builder(model,
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
					label_lst=None,
					output_type="sess",
					task_layer_reuse=None,
					**kargs):

	def model_fn(features, labels, mode):

		task_type = kargs.get("task_type", "cls")

		label_ids = features["{}_label_ids".format(task_type)]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		with tf.variable_scope(scope+"/{}/classifier".format(task_type), reuse=task_layer_reuse):
			(_, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob)

		loss_mask = tf.cast(features["{}_loss_multiplier".format(task_type)], tf.float32)
		masked_per_example_loss = per_example_loss * loss_mask
		loss = tf.reduce_sum(masked_per_example_loss) / (1e-10+tf.reduce_sum(loss_mask))

		if mode == tf.estimator.ModeKeys.TRAIN:
			multi_task_config = kargs.get("multi_task_config", {})
			if multi_task_config[task_type].get("lm_augumentation", False):
				print("==apply lm_augumentation==")
				masked_lm_positions = features["masked_lm_positions"]
				masked_lm_ids = features["masked_lm_ids"]
				masked_lm_weights = features["masked_lm_weights"]
				(masked_lm_loss,
				masked_lm_example_loss, 
				masked_lm_log_probs) = pretrain.get_masked_lm_output(
												model_config, 
												model.get_sequence_output(), 
												model.get_embedding_table(),
												masked_lm_positions, 
												masked_lm_ids, 
												masked_lm_weights,
												reuse=model_reuse)

				masked_lm_loss_mask = tf.expand_dims(loss_mask, -1) * tf.ones((1, multi_task_config[task_type]["max_predictions_per_seq"]))
				masked_lm_loss_mask = tf.reshape(masked_lm_loss_mask, (-1, ))

				masked_lm_label_weights = tf.reshape(masked_lm_weights, [-1])
				masked_lm_loss_mask *= tf.cast(masked_lm_label_weights, tf.float32)

				masked_lm_example_loss *= masked_lm_loss_mask# multiply task_mask
				masked_lm_loss = tf.reduce_sum(masked_lm_example_loss) / (1e-10+tf.reduce_sum(masked_lm_loss_mask))
				loss += multi_task_config[task_type]["masked_lm_loss_ratio"]*masked_lm_loss

				masked_lm_label_ids = tf.reshape(masked_lm_ids, [-1])
				
				print(masked_lm_log_probs.get_shape(), "===masked lm log probs===")
				print(masked_lm_label_ids.get_shape(), "===masked lm ids===")
				print(masked_lm_label_weights.get_shape(), "===masked lm mask===")

				lm_acc = build_accuracy(masked_lm_log_probs, masked_lm_label_ids, masked_lm_loss_mask)

		if kargs.get("task_invariant", "no") == "yes":
			print("==apply task adversarial training==")
			with tf.variable_scope(scope+"/dann_task_invariant", reuse=model_reuse):
				(_, 
				task_example_loss, 
				task_logits)  = distillation_utils.feature_distillation(model.get_pooled_output(), 
														1.0, 
														features["task_id"], 
														kargs.get("num_task", 7),
														dropout_prob, 
														True)
				masked_task_example_loss = loss_mask * task_example_loss
				masked_task_loss = tf.reduce_sum(masked_task_example_loss) / (1e-10+tf.reduce_sum(loss_mask))
				loss += kargs.get("task_adversarial", 1e-2) * masked_task_loss

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		if mode == tf.estimator.ModeKeys.TRAIN:
			multi_task_config = kargs.get("multi_task_config", {})
			if multi_task_config[task_type].get("lm_augumentation", False):
				print("==apply lm_augumentation==")
				masked_lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
												not_storage_params=not_storage_params)
				tvars.extend(masked_lm_pretrain_tvars)

		try:
			params_size = model_io_fn.count_params(model_config.scope)
			print("==total params==", params_size)
		except:
			print("==not count params==")
		# print(tvars)
		if load_pretrained == "yes":
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		if mode == tf.estimator.ModeKeys.TRAIN:

			acc = build_accuracy(logits, label_ids, loss_mask)

			return_dict = {
					"loss":loss, 
					"logits":logits,
					"task_num":tf.reduce_sum(loss_mask),
					"tvars":tvars
				}
			return_dict["{}_acc".format(task_type)] = acc
			if kargs.get("task_invariant", "no") == "yes":
				return_dict["{}_task_loss".format(task_type)] = masked_task_loss
				task_acc = build_accuracy(task_logits, features["task_id"], loss_mask)
				return_dict["{}_task_acc".format(task_type)] = task_acc
			if multi_task_config[task_type].get("lm_augumentation", False):
				return_dict["{}_masked_lm_loss".format(task_type)] = masked_lm_loss
				return_dict["{}_masked_lm_acc".format(task_type)] = lm_acc
			return return_dict
		elif mode == tf.estimator.ModeKeys.EVAL:
			eval_dict = {
				"loss":loss, 
				"logits":logits,
				"feature":model.get_pooled_output()
			}
			if kargs.get("adversarial", "no") == "adversarial":
				 eval_dict["task_logits"] = task_logits
			return eval_dict
	return model_fn


		

				