try:
	from .model_interface import model_zoo
except:
	from model_interface import model_zoo

import tensorflow as tf
import numpy as np
from bunch import Bunch

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from optimizer import distributed_optimizer as optimizer
from model_io import model_io

from distillation import knowledge_distillation as distill

def correlation(x, y):
	x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
	y = y - tf.reduce_mean(y, axis=-1, keepdims=True)
	x = tf.nn.l2_normalize(x, -1)
	y = tf.nn.l2_normalize(y, -1)
	return -tf.reduce_sum(x*y, axis=-1) # higher the better

def kd(x, y):
	x_prob = tf.nn.softmax(x)
	print(x_prob.get_shape(), y.get_shape(), tf.reduce_sum(x_prob * y, axis=-1).get_shape())
	return -tf.reduce_sum(x_prob * y, axis=-1) # higher the better

def mse(x, y):
	x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
	y = y - tf.reduce_mean(y, axis=-1, keepdims=True)
	return tf.reduce_sum((x-y)**2, axis=-1) # lower the better

def kd_distance(x, y, dist_type):
	if dist_type == "person":
		return correlation(x,y)
	elif dist_type == "kd":
		return kd(x, y)
	elif dist_type == "mse":
		return mse(x, y)

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

		model = model_api(model_config, features, labels,
							mode, target, reuse=model_reuse)

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
			(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob)
			label_loss = tf.reduce_sum(per_example_loss * features["label_ratio"]) / (1e-10+tf.reduce_sum(features["label_ratio"]))
			tf.get_variable_scope().reuse_variables()

			(tgt_loss, 
				tgt_per_example_loss, 
				tgt_logits) = classifier.classifier(model_config,
											features["distillation_feature"],
											num_labels,
											label_ids,
											dropout_prob)

		if mode == tf.estimator.ModeKeys.TRAIN:

			distillation_api = distill.KnowledgeDistillation(kargs.get("disitllation_config", Bunch({
														"logits_ratio_decay":"constant",
														"logits_ratio":0.5,
														"logits_decay_rate":0.999,
														"distillation":['mdd'],
														"feature_ratio":0.5,
														"feature_ratio_decay":"constant",
														"feature_decay_rate":0.999,
														"kd_type":"kd",
														"scope":scope
														})))
			# get teacher logits
			teacher_logit = tf.log(features["label_probs"]+1e-10)/kargs.get("temperature", 2.0) # log_softmax logits
			student_logit = tf.nn.log_softmax(logits /kargs.get("temperature", 2.0)) # log_softmax logits

			distillation_features = {
				"student_logits_tensor":student_logit,
				"teacher_logits_tensor":teacher_logit,
				"student_feature_tensor":model.get_pooled_output(),
				"teacher_feature_tensor":features["distillation_feature"],
				"student_label":tf.ones_like(label_ids, dtype=tf.int32),
				"teacher_label":tf.zeros_like(label_ids, dtype=tf.int32),
				"logits_ratio":kargs.get("logits_ratio", 0.5),
				"feature_ratio":kargs.get("logits_ratio", 0.5),
				"distillation_ratio":features["distillation_ratio"],
				"src_f_logit":logits,
				"tgt_f_logit":tgt_logits,
				"src_tensor":model.get_pooled_output(),
				"tgt_tensor":features["distillation_feature"]
			}

			distillation_loss = distillation_api.distillation(distillation_features,
										2, dropout_prob,
										model_reuse,
										opt_config.num_train_steps,
										feature_ratio=10,
										logits_ratio_decay="constant",
										feature_ratio_decay="constant",
										feature_decay_rate=0.999,
										logits_decay_rate=0.999,
										logits_ratio=0.5,
										scope=scope+"/adv_classifier",
										num_classes=num_labels,
										gamma=kargs.get("gamma", 4))

			loss = label_loss + distillation_loss["distillation_loss"]

		model_io_fn = model_io.ModelIO(model_io_config)

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

					try:
						pred_label = tf.argmax(distillation_loss["st_logits"], axis=-1, output_type=tf.int32)
						correct = tf.equal(
							tf.cast(tf.ones_like(label_ids, dtype=tf.int32), tf.int32),
							tf.cast(pred_label, tf.int32)
						)
						st_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

						pred_label = tf.argmax(distillation_loss["te_logits"], axis=-1, output_type=tf.int32)
						correct = tf.equal(
							tf.cast(tf.zeros_like(label_ids, dtype=tf.int32), tf.int32),
							tf.cast(pred_label, tf.int32)
						)
						te_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
					except:
						te_accuracy = tf.constant(0.0)
						st_accuracy = tf.constant(0.0)

					try:
						st_accuracy = tf.reduce_mean(distillation_loss["src_f1_prob"])						
						te_accuracy = tf.reduce_mean(distillation_loss["tgt_f1_prob"])
					except:
						te_accuracy = tf.constant(0.0)
						st_accuracy = tf.constant(0.0)

					return {
						"train":{
										"loss":loss, 
										"logits":logits,
										"train_op":train_op,
										"cross_entropy":label_loss,
										"distillation_loss":distillation_loss["distillation_loss"],
										"kd_num":tf.reduce_sum(features["distillation_ratio"]),
										"ce_num":tf.reduce_sum(features["label_ratio"]),
										"teacher_logit":teacher_logit,
										"student_logit":student_logit,
										"label_ratio":features["label_ratio"],
										"distilaltion_logits_loss":distillation_loss["distillation_logits_loss"],
										"distilaltion_feature_loss":distillation_loss["distillation_feature_loss"],
										"distillation_loss":distillation_loss["distillation_loss"],
										"st_accuracy":st_accuracy,
										"te_accuracy":te_accuracy,
										"mdd_loss":distillation_loss["mdd_loss"]
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

