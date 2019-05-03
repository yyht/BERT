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

from optimizer import distributed_optimizer as optimizer

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

		# model_api = model_zoo(model_config)

		# model = model_api(model_config, features, labels,
		# 					mode, target, reuse=model_reuse)

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

		task_mask = tf.cast(features["{}_mask".format(task_type)], tf.float32)

		masked_per_example_loss = task_mask * per_example_loss
		loss = tf.reduce_sum(masked_per_example_loss) / (1e-10+tf.reduce_sum(task_mask))

		if kargs.get("task_invariant", "no") == "yes":
			print("==apply task adversarial training==")
			with tf.variable_scope(scope+"/dann_task_invariant", reuse=model_reuse):
				(task_loss, 
				task_example_loss, 
				task_logits)  = distillation_utils.feature_distillation(model.get_pooled_output(), 
														1.0, 
														features["task_id"], 
														kargs.get("num_task", 7),
														dropout_prob, 
														True)
				loss += kargs.get("task_adversarial", 1e-2) * task_loss

		logits = tf.expand_dims(task_mask, axis=-1) * logits

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

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
			return {
					"loss":loss, 
					"logits":logits,
					"task_num":tf.reduce_sum(task_mask),
					"tvars":tvars
				}
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


		

				