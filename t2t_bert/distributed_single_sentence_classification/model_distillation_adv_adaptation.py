try:
	from .model_interface import model_zoo
except:
	from model_interface import model_zoo

import tensorflow as tf
import numpy as np

from utils.bert import bert_utils

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from optimizer import distributed_optimizer as optimizer
from model_io import model_io

from distillation.flip_gradient import flip_gradient

import copy

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

def adversarial_loss(model_config, feature, adv_ids, dropout_prob, model_reuse,
					**kargs):
	'''make the task classifier cannot reliably predict the task based on 
	the shared feature
	'''
	# input = tf.stop_gradient(input)
	feature = tf.nn.dropout(feature, 1 - dropout_prob)

	with tf.variable_scope(model_config.scope+"/adv_classifier", reuse=model_reuse):
		(adv_loss, 
			adv_per_example_loss, 
			adv_logits) = classifier.classifier(model_config,
										feature,
										kargs.get('adv_num_labels', 7),
										adv_ids,
										dropout_prob)
	return (adv_loss, adv_per_example_loss, adv_logits)

def diff_loss(shared_feat, task_feat):
	'''Orthogonality Constraints from https://github.com/tensorflow/models,
	in directory research/domain_adaptation
	'''
	task_feat -= tf.reduce_mean(task_feat, 0)
	shared_feat -= tf.reduce_mean(shared_feat, 0)

	task_feat = tf.nn.l2_normalize(task_feat, 1)
	shared_feat = tf.nn.l2_normalize(shared_feat, 1)

	correlation_matrix = tf.matmul(
		task_feat, shared_feat, transpose_a=True)

	cost = tf.reduce_mean(tf.square(correlation_matrix))
	cost = tf.where(cost > 0, cost, 0, name='value')

	assert_op = tf.Assert(tf.is_finite(cost), [cost])
	with tf.control_dependencies([assert_op]):
		loss_diff = tf.identity(cost)

	return loss_diff

def get_task_feature(config, common_feature, dropout_prob, scope):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		hidden_size = bert_utils.get_shape_list(common_feature, expected_rank=2)[-1]
		task_feature = tf.layers.dense(
						common_feature,
						hidden_size,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
		task_feature = tf.nn.dropout(task_feature, keep_prob=1 - dropout_prob)
		task_feature += common_feature
		task_feature = tf.layers.dense(
						task_feature,
						hidden_size,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
						activation=tf.tanh)
		return task_feature


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
							mode, target, reuse=tf.AUTO_REUSE)

		# model_adv_config = copy.deepcopy(model_config)
		# model_adv_config.scope = model_config.scope + "/adv_encoder"

		# model_adv_adaptation = model_api(model_adv_config, features, labels,
		# 					mode, target, reuse=tf.AUTO_REUSE)

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		common_feature = model.get_pooled_output()

		task_feature = get_task_feature(model_config, common_feature, dropout_prob, scope+"/task_residual")
		adv_task_feature = get_task_feature(model_config, flip_gradient(common_feature), dropout_prob, scope+"/adv_residual")

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			concat_feature = task_feature
			# concat_feature = tf.concat([task_feature, 
			# 							adv_task_feature], 
			# 							axis=-1)
			(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											concat_feature,
											num_labels,
											label_ids,
											dropout_prob)

		with tf.variable_scope(scope+"/adv_classifier", reuse=tf.AUTO_REUSE):
			adv_ids = features["adv_ids"]
			(adv_loss, 
				adv_per_example_loss, 
				adv_logits) = classifier.classifier(model_config,
											adv_task_feature,
											kargs.get('adv_num_labels', 12),
											adv_ids,
											dropout_prob)

		if mode == tf.estimator.ModeKeys.TRAIN:
			loss_diff = tf.constant(0.0)
			# adv_task_feature_no_grl = get_task_feature(model_config, common_feature, dropout_prob, scope+"/adv_residual")

			# loss_diff = diff_loss(task_feature, 
			# 						adv_task_feature_no_grl)

			print(kargs.get("temperature", 0.5), kargs.get("distillation_ratio", 0.5), "==distillation hyparameter==")

			# get teacher logits
			teacher_logit = tf.log(features["label_probs"]+1e-10)/kargs.get("temperature", 2.0) # log_softmax logits
			student_logit = tf.nn.log_softmax(logits /kargs.get("temperature", 2.0)) # log_softmax logits

			distillation_loss = kd_distance(teacher_logit, student_logit, kargs.get("distillation_distance", "kd")) 
			distillation_loss *= features["distillation_ratio"]
			distillation_loss = tf.reduce_sum(distillation_loss) / (1e-10+tf.reduce_sum(features["distillation_ratio"]))

			label_loss = tf.reduce_sum(per_example_loss * features["label_ratio"]) / (1e-10+tf.reduce_sum(features["label_ratio"]))
			print("==distillation loss ratio==", kargs.get("distillation_ratio", 0.9)*tf.pow(kargs.get("temperature", 2.0), 2))

			# loss = label_loss + kargs.get("distillation_ratio", 0.9)*tf.pow(kargs.get("temperature", 2.0), 2)*distillation_loss
			loss = (1-kargs.get("distillation_ratio", 0.9))*label_loss + kargs.get("distillation_ratio", 0.9) * distillation_loss
			if mode == tf.estimator.ModeKeys.TRAIN:
				loss += kargs.get("adv_ratio", 1.0) * adv_loss + loss_diff

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

					adv_pred_label = tf.argmax(adv_logits, axis=-1, output_type=tf.int32)
					adv_correct = tf.equal(
						tf.cast(adv_pred_label, tf.int32),
						tf.cast(adv_ids, tf.int32)
					)
					adv_accuracy = tf.reduce_mean(tf.cast(adv_correct, tf.float32))                 
					return {
						"train":{
										"loss":loss, 
										"logits":logits,
										"train_op":train_op,
										"cross_entropy":label_loss,
										"kd_loss":distillation_loss,
										"kd_num":tf.reduce_sum(features["distillation_ratio"]),
										"ce_num":tf.reduce_sum(features["label_ratio"]),
										"teacher_logit":teacher_logit,
										"student_logit":student_logit,
										"label_ratio":features["label_ratio"],
										"loss_diff":loss_diff,
										"adv_loss":adv_loss,
										"adv_accuracy":adv_accuracy
									},
						"hooks":training_hooks
					}
				elif output_type == "estimator":
					return estimator_spec

		elif mode == tf.estimator.ModeKeys.PREDICT:
			task_prob = tf.exp(tf.nn.log_softmax(logits))
			adv_prob = tf.exp(tf.nn.log_softmax(adv_logits))
			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'adv_prob':adv_prob,
												"task_prob":task_prob
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'adv_prob':adv_prob,
														"task_prob":task_prob
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

