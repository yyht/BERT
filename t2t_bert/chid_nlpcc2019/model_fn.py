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

# def window_span_classifier(config, features, sequence_output, 
# 					dropout_prob, hidden_dropout_prob,
# 					num_labels, mask_size):

# 	final_hidden_shape = bert_utils.get_shape_list(sequence_output, 
# 								expected_rank=3)

# 	answer_window_pos = features['label_window_positions']
# 	answer_window_shape = bert_utils.get_shape_list(answer_window_pos, 
# 								expected_rank=2)
# 	# (batch x (window_size x number_mask)) x dims
# 	input_window_tensor = bert_utils.gather_indexes(sequence_output, answer_window_pos)

# 	# (batch x number_mask) x (window_size x dims)
# 	input_window_tensor = tf.reshape(input_window_tensor,
# 											[final_hidden_shape[0]*mask_size,
# 											-1])

# 	intermediate_size = config.get('intermediate_size', 1024)
# 	hidden_size = config.get('hidden_size', 512)
# 	intermediate_act_fn = bert_modules.gelu

# 	# The activation is only applied to the "intermediate" hidden layer.
# 	with tf.variable_scope(config.scope+"/span/intermediate"):
# 		intermediate_output = tf.layers.dense(
# 				input_window_tensor,
# 				intermediate_size,
# 				activation=intermediate_act_fn,
# 				kernel_initializer=bert_modules.create_initializer(0.02))
# 		intermediate_output = bert_modules.dropout(intermediate_output, hidden_dropout_prob)
# 		intermediate_output = bert_modules.layer_norm(intermediate_output)

# 	# Down-project back to `hidden_size` then add the residual.
# 	with tf.variable_scope(config.scope+"/span/span_output"):
# 		layer_output = tf.layers.dense(
# 				intermediate_output,
# 				hidden_size,
# 				activation=intermediate_act_fn,
# 				kernel_initializer=bert_modules.create_initializer(0.02))
# 		layer_output = bert_modules.dropout(layer_output, hidden_dropout_prob)
# 		layer_output = bert_modules.layer_norm(layer_output)

# 	with tf.variable_scope(config.scope+"/span/predictions"): 

# 		output_weights = tf.get_variable(
# 			"output_weights", [num_labels, hidden_size],
# 			initializer=tf.truncated_normal_initializer(stddev=0.02))
	
# 		output_bias = tf.get_variable(
# 					"output_bias",
# 					shape=[num_labels],
# 					initializer=tf.zeros_initializer())
# 		logits = tf.matmul(layer_output, output_weights, transpose_b=True)
# 		logits = tf.nn.bias_add(logits, output_bias)

# 	label_ids = tf.reshape(features['label_ids'], [-1])
# 	label_weights = tf.reshape(features['label_weights'], [-1])

# 	if config.get("loss_type", "entropy") == "focal_loss":
# 		per_example_loss, _ = loss_utils.focal_loss_multi_v1(logits=logits, 
# 													labels=tf.stop_gradient(label_ids))
# 	else:
# 		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
# 													labels=tf.stop_gradient(label_ids),
# 													logits=logits)

# 	numerator = tf.reduce_sum(label_weights * per_example_loss)
# 	denominator = tf.reduce_sum(label_weights) + 1e-5
# 	loss = numerator / denominator

# 	return (loss, per_example_loss, logits)

def multi_position_classifier(config, features, 
		sequence_output,
		num_labels, dropout_prob):
	
	final_hidden_shape = bert_utils.get_shape_list(sequence_output, 
								expected_rank=3)

	print(final_hidden_shape, "====multi-choice shape====")

	answer_pos = tf.cast(features['label_positions'], tf.int32)
	cls_pos = tf.zeros_like(answer_pos)
	input_tensor = bert_utils.gather_indexes(sequence_output, answer_pos)
	cls_tensor = bert_utils.gather_indexes(sequence_output, cls_pos)

	answer_cls_tensor = tf.concat([cls_tensor, input_tensor], axis=-1)

	input_tensor = tf.layers.dense(
					answer_cls_tensor,
					units=config.hidden_size,
					activation=bert_modules.get_activation(config.hidden_act),
					kernel_initializer=bert_modules.create_initializer(
							config.initializer_range))
	input_tensor = bert_modules.layer_norm(input_tensor)

	output_weights = tf.get_variable(
			"output_weights", [num_labels, final_hidden_shape[-1]],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
				"output_bias",
				shape=[num_labels],
				initializer=tf.zeros_initializer())
	logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	label_ids = tf.reshape(tf.cast(features['label_ids'], tf.int32), [-1])
	label_weights = tf.reshape(tf.cast(features['label_weights'], tf.float32), [-1])

	if config.get('class_weights', None):
		class_weights = tf.constant(np.array(config.class_weights).astype(np.float32))
		
	if config.get("loss", "entropy") == "focal_loss":
		per_example_loss, _ = loss_utils.focal_loss_multi_v1(config,logits=logits, 
													labels=tf.stop_gradient(label_ids))
	elif config.get("loss", "smoothed_ce") == 'smoothed_ce':
		per_example_loss = loss_utils.ce_label_smoothing(config, logits=logits,
														labels=tf.stop_gradient(label_ids))
	elif config.get('loss', 'class_balanced_focal') == 'class_balanced_focal':
		per_example_loss, _ = loss_utils.class_balanced_focal_loss_multi_v1(
											config, logits=logits, 
											labels=label_ids, 
											label_weights=class_weights)
	else:
		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													labels=tf.stop_gradient(label_ids),
													logits=logits)

	numerator = tf.reduce_sum(label_weights * per_example_loss)
	denominator = tf.reduce_sum(label_weights) + 1e-5
	loss = numerator / denominator

	return (loss, per_example_loss, logits)

def eval_logtis(logits, 
		features,
		num_labels):

	label_ids = tf.reshape(tf.cast(features['label_ids'], tf.int32), [-1])
	label_weights = tf.reshape(tf.cast(features['label_weights'], tf.int32), [-1])

	sentence_predictions = tf.argmax(
					logits, axis=-1, output_type=tf.int32)

	sentence_accuracy = tf.metrics.accuracy(
					labels=label_ids, predictions=sentence_predictions,
					weights=label_weights)

	sentence_f = tf_metrics.f1(label_ids, 
							sentence_predictions, 
							num_labels, 
							weights=label_weights, 
							average="macro")

	eval_metric_ops = {
									"f1": sentence_f,
									"acc":sentence_accuracy
							}

	return eval_metric_ops

def train_metric_fn(logits, 
		features,
		num_labels):

	label_ids = tf.reshape(tf.cast(features['label_ids'], tf.int32), [-1])
	label_weights = tf.reshape(tf.cast(features['label_weights'], tf.float32), [-1])

	sentence_predictions = tf.argmax(
					logits, axis=-1, output_type=tf.int32)

	sentence_accuracy = tf.equal(
						tf.cast(sentence_predictions, tf.int32),
						tf.cast(label_ids, tf.int32))
	sentence_accuracy = tf.cast(sentence_accuracy, tf.float32)

	sentence_accuracy = tf.reduce_sum(sentence_accuracy*label_weights)/tf.reduce_sum(label_weights+1e-15)

	train_metric = {
						"train_acc":sentence_accuracy,
						"label_weights":tf.reduce_sum(label_weights),
						"label_ids":tf.reduce_sum(label_ids)
					}

	return train_metric
	

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
	
	# model_config.class_weights = [0.7, 1.3]
	model_config.loss = 'focal_loss'

	print("=apply chid pair match==")

	def model_fn(features, labels, mode):

		features['input_mask'] = tf.cast(tf.not_equal(features['input_ids'], 
													kargs.get('[PAD]', 0)), tf.int64)

		# for key in ['input_mask', 'input_ids', 'segment_ids']:
		# 	features[key] = features[key][:, :274]

		model = bert_encoder(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE)

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
			logits) = multi_position_classifier(model_config, 
								features,
								model.get_sequence_output(),
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

			# train_op, hooks = model_io_fn.get_ema_hooks(train_op, 
			# 				tvars,
			# 				kargs.get('params_moving_average_decay', 0.99),
			# 				scope, mode, 
			# 				first_stage_steps=opt_config.num_warmup_steps,
			# 				two_stage=True)

			model_io_fn.set_saver()

			train_metric_dict = train_metric_fn(
						logits, 
					features,
					num_labels
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
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			max_prob = tf.reduce_max(prob, axis=-1)

			# _, hooks = model_io_fn.get_ema_hooks(None,
			# 							None,
			# 							kargs.get('params_moving_average_decay', 0.99), 
			# 							scope, mode)
			
			hooks = []

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
									},
									prediction_hooks=hooks

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
							num_labels)
			
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops,
								evaluation_hooks=eval_hooks)
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

