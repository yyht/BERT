from distributed_encoder.bert_encoder import bert_encoder

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from model_io import model_io
from utils.bert import bert_utils, bert_modules
from loss import loss_utils
from metric import tf_metrics
import math

from tensorflow.contrib.crf import crf_log_likelihood

NAN = 1

def zero_transition(shape):
	transition = tf.zeros((shape[1], shape[1]))
	transition = transition - tf.eye(shape[1])*NAN
	return tf.cast(transition, tf.float32)

def multi_position_crf_classifier(config, features, 
		model_dict, num_labels, dropout_prob):

	batch_size = features['batch_size']
	total_length_a = features['total_length_a']
	total_length_b = features['total_length_b']

	sequence_output_a = model_dict["a"].get_sequence_output() # [batch x 10, 130, 768]
	shape_lst = bert_utils.get_shape_list(sequence_output_a, 
								expected_rank=3)

	sequence_output_a = tf.reshape(sequence_output_a, [-1, total_length_a, shape_lst[-1]]) # [batch, 10 x 130, 768]
	answer_pos = tf.cast(features['label_positions'], tf.int32)
	sequence_output_a = bert_utils.gather_indexes(sequence_output_a, answer_pos) # [batch*10, 768]

	sequence_output_a = tf.reshape(sequence_output_a, [-1, config.max_predictions_per_seq, 
														shape_lst[-1]]) # [batch, 10, 768]

	sequence_output_b = model_dict["b"].get_pooled_output() # [batch x 10,768]
	sequence_output_b = tf.reshape(sequence_output_b, [-1, num_labels, shape_lst[-1]])  # [batch, 10, 768]
	seq_b_shape = bert_utils.get_shape_list(sequence_output_b, 
								expected_rank=3)
	
	cross_matrix = tf.get_variable(
			"output_weights", [shape_lst[-1], shape_lst[-1]],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	# batch x 10 x 768
	sequence_output_a_proj = tf.einsum("abc,cd->abd", sequence_output_a, cross_matrix)

	# batch x 10 x 768. batch x 10 x 768
	# batch x 10(ans_pos) x 11(ans_field)
	logits = tf.einsum("abd,acd->abc", sequence_output_a_proj, sequence_output_b)
	logits = tf.multiply(logits, 1.0 / tf.math.sqrt(tf.cast(shape_lst[-1], tf.float32)))

	

	# print(sequence_output_a.get_shape(), sequence_output_b.get_shape(), logits.get_shape())

	# label_ids = tf.cast(features['label_ids'], tf.int32)
	# label_weights = tf.cast(features['label_weights'], tf.int32)
	# label_seq_length = tf.reduce_sum(label_weights, axis=-1)

	# transition = zero_transition(seq_b_shape)

	# log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
	# 										inputs=logits,
	# 										tag_indices=label_ids,
	# 										sequence_lengths=label_seq_length,
	# 										transition_params=transition)

	# transition_params = tf.stop_gradient(transition_params)
	# per_example_loss = -log_likelihood
	# loss = tf.reduce_mean(per_example_loss)

	return (loss, per_example_loss, logits, transition_params)

def eval_logtis(logits, 
		features,
		num_labels, transition_params):

	label_ids = tf.cast(features['label_ids'], tf.int32)
	label_weights = tf.cast(features['label_weights'], tf.int32)
	label_seq_length = tf.reduce_sum(label_weights, axis=-1)

	label_ids = tf.reshape(tf.cast(label_ids, tf.int32), [-1])
	label_weights = tf.reshape(tf.cast(label_weights, tf.int32), [-1])

	decode_tags, best_score = tf.contrib.crf.crf_decode(
				logits,
				transition_params,
				label_seq_length)

	decode_tags = tf.reshape(tf.cast(decode_tags, tf.int32), [-1])
	sentence_accuracy = tf.metrics.accuracy(
					labels=label_ids, 
					predictions=decode_tags,
					weights=tf.cast(label_weights, tf.float32))

	eval_metric_ops = {
		"acc":sentence_accuracy
	}

	return eval_metric_ops

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

	model_config.max_predictions_per_seq = 10
	
	def model_fn(features, labels, mode):

		shape_lst_a = bert_utils.get_shape_list(features['input_ids_a'])
		batch_size_a = shape_lst_a[0]
		total_length_a = shape_lst_a[1]

		shape_lst_b = bert_utils.get_shape_list(features['input_ids_b'])
		batch_size_b = shape_lst_b[0]
		total_length_b = shape_lst_b[1]

		features['input_ids_a'] = tf.reshape(features['input_ids_a'], 
										[-1, model_config.max_length])
		features['segment_ids_a'] = tf.reshape(features['segment_ids_a'], 
										[-1, model_config.max_length])
		features['input_mask_a'] = tf.cast(tf.not_equal(features['input_ids_a'], 
											kargs.get('[PAD]', 0)), tf.int64)

		features['input_ids_b'] = tf.reshape(features['input_ids_b'], 
										[-1, model_config.max_predictions_per_seq])
		features['segment_ids_b'] = tf.reshape(features['segment_ids_b'], 
										[-1, model_config.max_predictions_per_seq])
		features['input_mask_b'] = tf.cast(tf.not_equal(features['input_ids_b'], 
											kargs.get('[PAD]', 0)), tf.int64)

		features['batch_size'] = batch_size_a
		features['total_length_a'] = total_length_a
		features['total_length_b'] = total_length_b

		model_dict = {}
		for target in ["a", "b"]:
			model = bert_encoder(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE)
			model_dict[target] = model
		
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
			logits,
			transition_params) = multi_position_crf_classifier(
											model_config, 
											features, 
											model_dict, 
											num_labels, 
											dropout_prob)

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

			train_op, hooks = model_io_fn.get_ema_hooks(train_op, 
							tvars,
							kargs.get('params_moving_average_decay', 0.99),
							scope, mode, 
							first_stage_steps=opt_config.num_warmup_steps,
							two_stage=True)

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
			print(tf.global_variables(), "==global_variables==")
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

			label_weights = tf.cast(features['label_weights'], tf.int32)
			label_seq_length = tf.reduce_sum(label_weights, axis=-1)
			
			decode_tags, best_score = tf.contrib.crf.crf_decode(
								logits,
								transition_params,
								label_seq_length
			)

			_, hooks = model_io_fn.get_ema_hooks(None,
										None,
										kargs.get('params_moving_average_decay', 0.99), 
										scope, mode)
			
			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'decode_tags':decode_tags,
												"best_score":best_score,
												"transition_params":transition_params,
												"logits":logits
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'decode_tags':decode_tags,
														"best_score":best_score,
														"transition_params":transition_params,
														"logits":logits
													}
												)
									},
									prediction_hooks=[hooks]

						)
			return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:

			_, hooks = model_io_fn.get_ema_hooks(None,
										None,
										kargs.get('params_moving_average_decay', 0.99), 
										scope, mode)
			eval_hooks = []

			if output_type == "sess":
				return {
					"eval":{
							"per_example_loss":per_example_loss,
							"logits":logits,
							"loss":tf.reduce_mean(per_example_loss),
							"feature":model.get_pooled_output()
						}
				}
			elif output_type == "estimator":

				eval_metric_ops = eval_logtis(
							logits, 
							features,
							num_labels,
							transition_params)
			
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops,
								evaluation_hooks=eval_hooks)
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

