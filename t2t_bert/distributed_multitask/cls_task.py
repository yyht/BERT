try:
	from .model_interface import model_zoo
except:
	from model_interface import model_zoo

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from optimizer import distributed_optimizer as optimizer
from model_io import model_io

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
					label_lst=None,
					output_type="sess",
					task_layer_reuse=None,
					**kargs):

	def model_fn(features, labels, mode):

		model_api = model_zoo(model_config)

		model = model_api(model_config, features, labels,
							mode, target, reuse=model_reuse)

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

		logits = tf.expand_dims(task_mask, axis=-1) * logits

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		try:
			params_size = model_io_fn.count_params(model_config.scope)
			print("==total params==", params_size)
		except:
			print("==not count params==")
		print(tvars)
		if load_pretrained == "yes":
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		if mode == tf.estimator.ModeKeys.TRAIN:

			optimizer_fn = optimizer.Optimizer(opt_config)

			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print("==update_ops==", update_ops)
			with tf.control_dependencies(update_ops):
				train_op = optimizer_fn.get_train_op(loss, tvars, 
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

				return {
					"loss":loss, 
					"logits":logits,
					"train_op":train_op,
					"training_hooks":training_hooks,
					"task_num":tf.reduce_sum(task_mask)
				}
		elif mode == tf.estimator.ModeKeys.EVAL:
			return {
				"loss":loss, 
				"logits":logits,
				"feature":model.get_pooled_output()
			}
	return model_fn