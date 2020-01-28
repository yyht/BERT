try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from optimizer import distributed_optimizer as optimizer
from model_io import model_io

class ModelFnBuilder(object):
	def __init__(self, model_config,
					num_labels,
					init_checkpoint,
					load_pretrained=True,
					model_io_config={},
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="a",
					label_lst=None,
					output_type="sess",
					**kargs):
		self.model_config = model_config
		self.num_labels = num_labels
		self.init_checkpoint = init_checkpoint
		self.load_pretrained = load_pretrained
		self.model_io_config = model_io_config
		self.opt_config = opt_config
		self.exclude_scope = exclude_scope
		self.not_storage_params = not_storage_params
		self.target = target
		self.label_lst = label_lst
		self.output_type = output_type
		self.kargs = kargs
		self.model_io_fn = model_io.ModelIO(self.model_io_config)
		self.optimizer_fn = optimizer.Optimizer(self.opt_config)

	def model_fn(self, features, labels, model_reuse):
		model_api = model_zoo(self.model_config)

		model_lst = []

		assert len(self.target.split(",")) == 2
		target_name_lst = self.target.split(",")
		print(target_name_lst)
		for index, name in enumerate(target_name_lst):
			if index > 0:
				reuse = True
			else:
				reuse = model_reuse
			model_lst.append(model_api(self.model_config, features, labels,
							mode, name, reuse=reuse))

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = self.model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if self.model_io_config.fix_lm == True:
			scope = self.model_config.scope + "_finetuning"
		else:
			scope = self.model_config.scope

		with tf.variable_scope(scope, reuse=self.model_reuse):
			seq_output_lst = [model.get_pooled_output() for model in model_lst]
			if self.model_config.get("classifier", "order_classifier") == "order_classifier":
				[loss, 
					per_example_loss, 
					logits] = classifier.order_classifier(
								model_config, seq_output_lst, 
								num_labels, label_ids,
								dropout_prob, ratio_weight=None)
			elif model_config.get("classifier", "order_classifier") == "siamese_interaction_classifier":
				[loss, 
					per_example_loss, 
					logits] = classifier.siamese_classifier(
								model_config, seq_output_lst, 
								self.num_labels, label_ids,
								dropout_prob, ratio_weight=None)

		params_size = self.model_io_fn.count_params(self.model_config.scope)
		print("==total params==", params_size)

		self.tvars = model_io_fn.get_params(self.model_config.scope, 
										not_storage_params=self.not_storage_params)
		print(tvars)
		if self.load_pretrained == "yes":
			self.model_io_fn.load_pretrained(self.tvars, 
										self.init_checkpoint,
										exclude_scope=self.exclude_scope)
		self.loss = loss
		self.per_example_loss = per_example_loss
		self.logits = logits

		return self.loss

	def get_train_op(self, features, labels, mode, model_reuse):

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			if self.opt_config.get("train_op", "pai_soar") == "pai_soar":
				self.train_op = self.optimizer_fn.get_train_op(
							self.model_fn(features, labels, model_reuse), 
							[], 
							self.opt_config.init_lr, 
							self.opt_config.num_train_steps,
							self.kargs)
			else:
				loss = self.model_fn(features, labels, mode, model_reuse)
				self.train_op = self.optimizer_fn.get_train_op(
							loss, 
							self.tvars, 
							self.opt_config.init_lr, 
							self.opt_config.num_train_steps,
							self.kargs)

			self.model_io_fn.set_saver()

			if self.kargs.get("task_index", 1) == 0 and self.kargs.get("run_config", None):
				training_hooks = []
			elif self.kargs.get("task_index", 1) == 0:
				self.model_io_fn.get_hooks(self.kargs.get("checkpoint_dir", None), 
													self.kargs.get("num_storage_steps", 1000))

				training_hooks = self.model_io_fn.checkpoint_hook
			else:
				training_hooks = []

			if len(self.optimizer_fn.distributed_hooks) >= 1:
				training_hooks.extend(self.optimizer_fn.distributed_hooks)
			print(training_hooks, "==training_hooks==", "==task_index==", kargs.get("task_index", 1))

			estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
							loss=loss, train_op=train_op,
							training_hooks=training_hooks)
			if output_type == "sess":
				if self.opt_config.get("train_op", "pai_soar") == "pai_soar":
					return {
						"train":{
										"loss":loss, 
										"train_op":train_op
									},
						"hooks":training_hooks
					}
			elif output_type == "estimator":
				return estimator_spec


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
					**kargs):

	def model_fn(features, labels, mode):

		model_api = model_zoo(model_config)

		model_lst = []

		assert len(target.split(",")) == 2
		target_name_lst = target.split(",")
		print(target_name_lst)
		for index, name in enumerate(target_name_lst):
			if index > 0:
				reuse = True
			else:
				reuse = model_reuse
			model_lst.append(model_api(model_config, features, labels,
							mode, name, reuse=reuse))

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		with tf.variable_scope(scope, reuse=model_reuse):
			seq_output_lst = [model.get_pooled_output() for model in model_lst]
			if model_config.get("classifier", "order_classifier") == "order_classifier":
				[loss, 
					per_example_loss, 
					logits] = classifier.order_classifier(
								model_config, seq_output_lst, 
								num_labels, label_ids,
								dropout_prob, ratio_weight=None)
			elif model_config.get("classifier", "order_classifier") == "siamese_interaction_classifier":
				[loss, 
					per_example_loss, 
					logits] = classifier.siamese_classifier(
								model_config, seq_output_lst, 
								num_labels, label_ids,
								dropout_prob, ratio_weight=None)

		model_io_fn = model_io.ModelIO(model_io_config)

		params_size = model_io_fn.count_params(model_config.scope)
		print("==total params==", params_size)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)
		print(tvars)
		if load_pretrained == "yes":
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		if mode == tf.estimator.ModeKeys.TRAIN:

			optimizer_fn = optimizer.Optimizer(opt_config)

			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss, train_op=train_op,
								training_hooks=training_hooks)
				if output_type == "sess":
					return {
						"train":{
										"loss":loss, 
										"logits":logits,
										"train_op":train_op
									},
						"hooks":training_hooks
					}
				elif output_type == "estimator":
					return estimator_spec

		elif mode == tf.estimator.ModeKeys.PREDICT:
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			max_prob = tf.reduce_max(prob, axis=-1)
			
			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'pred_label':pred_label,
												"max_prob":max_prob
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'pred_label':pred_label,
														"max_prob":max_prob
													}
												)
									}
						)
			return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(per_example_loss,
						logits, 
						label_ids):
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
										num_labels, 
										label_lst, average="macro")

				eval_metric_ops = {
									"f1": sentence_f,
									"acc":sentence_accuracy
								}

				return eval_metric_ops

			eval_metric_ops = metric_fn( 
							per_example_loss,
							logits, 
							label_ids)
			
			estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops)

			if output_type == "sess":
				return {
					"eval":{
							"per_example_loss":per_example_loss,
							"logits":logits,
							"loss":tf.reduce_mean(per_example_loss)
						}
				}
			elif output_type == "estimator":
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn


