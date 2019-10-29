import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

import tensorflow as tf
import numpy as np
from optimizer import distributed_optimizer as optimizer
from model_io import model_io

from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

def train_metric_fn(masked_lm_example_loss, masked_lm_log_probs, 
					masked_lm_ids,
					masked_lm_weights, 
					next_sentence_example_loss,
					next_sentence_log_probs, 
					next_sentence_labels):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])

	masked_lm_accuracy = tf.equal(
						tf.cast(masked_lm_ids, tf.int32),
						tf.cast(masked_lm_predictions, tf.int32)
					)
	masked_lm_accuracy = tf.cast(masked_lm_accuracy, tf.int32)*tf.cast(masked_lm_weights, dtype=tf.int32)
	masked_lm_accuracy = tf.reduce_sum(tf.cast(masked_lm_accuracy, tf.float32)) / tf.reduce_sum(masked_lm_weights)
	masked_lm_mean_loss = tf.reduce_sum(masked_lm_example_loss*masked_lm_weights) / tf.reduce_sum(masked_lm_weights)

	next_sentence_log_probs = tf.reshape(
			next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
	next_sentence_predictions = tf.argmax(
			next_sentence_log_probs, axis=-1, output_type=tf.int32)
	next_sentence_labels = tf.reshape(next_sentence_labels, [-1])

	next_sentence_accuracy = tf.equal(
						tf.cast(next_sentence_labels, tf.int32),
						tf.cast(next_sentence_predictions, tf.int32)
					)
	next_sentence_accuracy = tf.reduce_mean(tf.cast(next_sentence_accuracy, tf.float32))
	next_sentence_loss = tf.reduce_mean(next_sentence_example_loss)

	return {
		"masked_lm_accuracy": masked_lm_accuracy,
		"masked_lm_loss": masked_lm_mean_loss,
		"next_sentence_accuracy": next_sentence_accuracy,
		"next_sentence_loss": next_sentence_loss,
		"valid_position":tf.reduce_sum(masked_lm_weights)
		}

def classifier_model_fn_builder(
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

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		(nsp_loss, 
		nsp_per_example_loss, 
		nsp_log_prob) = pretrain.get_next_sentence_output(model_config,
										model.get_pooled_output(),
										features['next_sentence_labels'],
										reuse=model_reuse)

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
		loss = model_config.lm_ratio * masked_lm_loss + model_config.nsp_ratio * nsp_loss
		
		model_io_fn = model_io.ModelIO(model_io_config)

		if mode == tf.estimator.ModeKeys.TRAIN:
			pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

			lm_pretrain_tvars = model_io_fn.get_params("cls", 
										not_storage_params=not_storage_params)

			pretrained_tvars.extend(lm_pretrain_tvars)

			optimizer_fn = optimizer.Optimizer(opt_config)
			
			if load_pretrained:
				model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope)

			tvars = pretrained_tvars
			model_io_fn.print_params(tvars, string=", trainable params")
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):

				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				train_op, hooks = model_io_fn.get_ema_hooks(train_op, 
									tvars,
									kargs.get('params_moving_average_decay', 0.99),
									scope, mode, 
									first_stage_steps=opt_config.num_warmup_steps,
									two_stage=True)

				model_io_fn.set_saver()

				train_metric_dict = train_metric_fn(
						masked_lm_example_loss, masked_lm_log_probs, 
						masked_lm_ids,
						masked_lm_weights, 
						nsp_per_example_loss,
						nsp_log_prob, 
						features['next_sentence_labels']
					)

				for key in train_metric_dict:
					tf.summary.scalar(key, train_metric_dict[key])
				tf.summary.scalar('learning_rate', optimizer_fn.learning_rate)

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
								loss=loss, train_op=train_op)

				if output_type == "sess":
					return {
						"train":{
										"loss":loss, 
										"nsp_log_pro":nsp_log_prob,
										"train_op":train_op,
										"masked_lm_loss":masked_lm_loss,
										"next_sentence_loss":nsp_loss,
										"masked_lm_log_pro":masked_lm_log_probs
									},
						"hooks":training_hooks
					}
				elif output_type == "estimator":
					return estimator_spec

		elif mode == tf.estimator.ModeKeys.PREDICT:

			def prediction_fn(logits):

				predictions = {
					"nsp_classes": tf.argmax(input=nsp_log_prob, axis=1),
					"nsp_probabilities": 
						tf.exp(nsp_log_prob, name="nsp_softmax"),
					"masked_vocab_classes":tf.argmax(input=masked_lm_log_probs, axis=1),
					"masked_probabilities":tf.exp(masked_lm_log_probs, name='masked_softmax')
				}
				return predictions

			predictions = prediction_fn(logits)

			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions=predictions,
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													predictions
												)
									}
						)
			return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:

			def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
					masked_lm_weights, next_sentence_example_loss,
					next_sentence_log_probs, next_sentence_labels):
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

				next_sentence_log_probs = tf.reshape(
					next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
				next_sentence_predictions = tf.argmax(
					next_sentence_log_probs, axis=-1, output_type=tf.int32)
				next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
				next_sentence_accuracy = tf.metrics.accuracy(
					labels=next_sentence_labels, predictions=next_sentence_predictions)
				next_sentence_mean_loss = tf.metrics.mean(
					values=next_sentence_example_loss)

				return {
					"masked_lm_accuracy": masked_lm_accuracy,
					"masked_lm_loss": masked_lm_mean_loss,
					"next_sentence_accuracy": next_sentence_accuracy,
					"next_sentence_loss": next_sentence_mean_loss
					}

			if output_type == "sess":
				return {
					"eval":{
							"nsp_log_prob":nsp_log_prob,
							"masked_lm_log_prob":masked_lm_log_probs,
							"nsp_loss":nsp_loss,
							"masked_lm_loss":masked_lm_loss,
							"feature":model.get_pooled_output()
						}
				}
			elif output_type == "estimator":
				eval_metric_ops = metric_fn(masked_lm_example_loss, 
											masked_lm_log_probs, 
											masked_lm_ids,
											masked_lm_weights, 
											nsp_per_example_loss,
											nsp_log_prob, 
											features['next_sentence_labels'])
				_, hooks = model_io_fn.get_ema_hooks(None,
										None,
										kargs.get('params_moving_average_decay', 0.99), 
										scope, mode)

				eval_hooks = [hooks] if hooks else []

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops,
								evaluation_hooks=eval_hooks)
				return estimator_spec
		else:
			raise NotImplementedError()

	return model_fn