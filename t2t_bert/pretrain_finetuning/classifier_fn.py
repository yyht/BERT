import tensorflow as tf
import numpy as np

from model.bert import bert
from model_io import model_io
try:
	from optimizer import hvd_distributed_optimizer as optimizer
except:
	from optimizer import optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils

from metric import tf_metrics


def base_model(model_config, features, labels, 
			mode, reuse=None):
	
	input_ids = features["input_ids"]
	input_mask = features["input_mask"]
	segment_ids = features["segment_ids"]

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = bert.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse,
						perturbation=None)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_pooler(reuse=reuse)

	return model

def classifier_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							optimizer_fn=None,
							model_io_config={},
							opt_config={},
							exclude_scope="",
							not_storage_params=[],
							label_lst=None):

	def model_fn(features, labels, mode):

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			hidden_dropout_prob = model_config.hidden_dropout_prob
			attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
			dropout_prob = model_config.dropout_prob
		else:
			hidden_dropout_prob = 0.0
			attention_probs_dropout_prob = 0.0
			dropout_prob = 0.0

		model = base_model(model_config, features, labels, 
			mode, reuse=reuse)

		with tf.variable_scope(model_config.scope, reuse=reuse):
			(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob)

		if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
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
											reuse=reuse)
			total_loss = model_config.lm_ratio * masked_lm_loss + loss
		else:
			total_loss = loss

		if mode == tf.estimator.ModeKeys.TRAIN:
			pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

			masked_lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
										not_storage_params=not_storage_params)

			pretrained_tvars.extend(masked_lm_pretrain_tvars)
			
			if load_pretrained:
				model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope)

			tvars = pretrained_tvars
			model_io_fn.print_params(tvars, string=", trainable params")
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				# optimizer_fn = optimizer.Optimizer(opt_config)
				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				output_dict = {"train_op":train_op,
							"total_loss":total_loss,
							"masked_lm_loss":masked_lm_loss,
							"sentence_loss":loss}

				return output_dict

		elif mode == tf.estimator.ModeKeys.PREDICT:

			def prediction_fn(logits):

				predictions = {
					"classes": tf.argmax(input=logits, axis=1),
					"probabilities": 
						tf.exp(tf.nn.log_softmax(logits, name="softmax_tensor"))
				}
				return predictions

			predictions = prediction_fn(logits)

			return predictions

		elif mode == tf.estimator.ModeKeys.EVAL:

			def metric_fn(masked_lm_example_loss, masked_lm_log_probs, 
					masked_lm_ids,
					masked_lm_weights, 
					per_example_loss,
					logits, 
					label_ids):
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

				return {
					"masked_lm_accuracy": masked_lm_accuracy,
					"masked_lm_loss": masked_lm_mean_loss,
					"sentence_f": sentence_f,
					"sentence_loss": sentence_mean_loss,
					"probabilities":tf.exp(tf.nn.log_softmax(logits, name="softmax_tensor")),
					"label_ids":label_ids
					}

			eval_metric_ops = metric_fn(masked_lm_example_loss, 
							masked_lm_log_probs, 
							masked_lm_ids,
							masked_lm_weights, 
							per_example_loss,
							logits, 
							label_ids)
			
			return eval_metric_ops

	return model_fn