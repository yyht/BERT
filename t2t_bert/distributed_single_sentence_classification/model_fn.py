try:
	from .model_interface import model_zoo
except:
	from model_interface import model_zoo

import tensorflow as tf
import numpy as np
from utils.bert import bert_utils
from loss import loss_utils, triplet_loss_utils

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from optimizer import distributed_optimizer as optimizer
from model_io import model_io
from utils.simclr import simclr_utils

def get_labels_of_similarity(query_input_ids, anchor_query_ids):
	idxs_1 = tf.expand_dims(query_input_ids, axis=1) # batch 1 seq
	idxs_2 = tf.expand_dims(anchor_query_ids, axis=0) # 1 batch seq
	# batch x batch x seq
	labels = tf.cast(tf.not_equal(idxs_1, idxs_2), tf.float32) # not equal:1, equal:0
	equal_num = tf.reduce_sum(labels, axis=-1) # [batch, batch]
	not_equal_label = tf.cast(tf.not_equal(equal_num, 0), tf.float32)
	not_equal_label_shape = bert_utils.get_shape_list(not_equal_label, expected_rank=[2,3])
	not_equal_label *= tf.cast(1 - tf.eye(not_equal_label_shape[0]), tf.float32) 
	equal_label = (1 - not_equal_label) - tf.eye(not_equal_label_shape[0])
	return equal_label, not_equal_label

def get_finised_pos_v1(token_seq, finished_index, max_length): 
	token_seq = tf.cast(token_seq, tf.int32)
	seq_shape = bert_utils.get_shape_list(token_seq, expected_rank=[2,3])
	match_indices = tf.where(                          # [[5, 5, 2, 5, 4],
	tf.equal(finished_index, token_seq),                              #  [0, 5, 2, 3, 5],
		x=tf.range(seq_shape[1]) * tf.ones_like(token_seq),  #  [5, 1, 5, 5, 5]]
		y=(seq_shape[1])*tf.ones_like(token_seq))

	finished_pos = tf.reduce_min(match_indices, axis=1)
	mask = tf.cast(tf.one_hot(finished_pos, max_length), tf.float32) # [batch, max_length]
				
	modified_token_seq = tf.cast(token_seq, tf.float32) - float(finished_index) * mask + mask * 1.0
				
	return tf.cast(modified_token_seq, tf.int32)


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

		# if kargs.get('trf_input', False):

		# 	input_shape_list = bert_utils.get_shape_list(features['input_ids'], expected_rank=2)
		# 	batch_size = input_shape_list[0]
		# 	seq_length = input_shape_list[1]

		# 	input_ids = get_finised_pos_v1(features['input_ids'], 102, seq_length)
		# 	features['input_ids'] = input_ids

		# 	tf.logging.info("**** trf input modification for qa and sentence pair **** ")

		# features['segment_ids'] = 0 * features['segment_ids']
		model = model_api(model_config, features, labels,
							mode, target, reuse=model_reuse, **kargs)

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
			is_training = True
		else:
			dropout_prob = 0.0
			is_training = False

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

		if not kargs.get('use_tpu'):
			tf.summary.scalar("classifier_loss", loss)
				
		cpc_flag = model_config.get('apply_cpc', 'none')
		if model_config.get("label_type", "single_label") == "multi_label":
			pos_true_mask, neg_true_mask = get_labels_of_similarity(label_ids, label_ids)
		else:
			label_shape = bert_utils.get_shape_list(label_ids, expected_rank=[1, 2,3])
			neg_true_mask = tf.cast(triplet_loss_utils._get_anchor_negative_triplet_mask(label_ids), tf.float32)
			pos_true_mask = (1.0 - neg_true_mask)*(1.0-tf.eye(label_shape[0], dtype=tf.float32))
		if cpc_flag == 'apply':
			loss_mask = tf.minimum(tf.reduce_sum(pos_true_mask, axis=-1), 1)
			feat = model.get_pooled_output()

			if kargs.get("apply_feat_projection", False):
				with tf.variable_scope(scope+"/head_proj", reuse=tf.AUTO_REUSE):
					feat = simclr_utils.projection_head(feat, 
											is_training, 
											head_proj_dim=128,
											num_nlh_layers=1,
											head_proj_mode='nonlinear',
											name='head_contrastive')
				feat = tf.nn.l2_normalize(feat+1e-20, axis=-1)
				cosine_score = tf.matmul(feat, tf.transpose(feat)) / 0.1
				tf.logging.info("****** apply simclr projection and. l2 normalize *******")
			else:
				cosine_score = tf.matmul(feat, tf.transpose(feat))
				tf.logging.info("****** apply raw feat *******")
			cosine_score_neg = neg_true_mask * cosine_score
			cosine_score_pos = -pos_true_mask * cosine_score

			y_pred_neg = cosine_score_neg - (1 - neg_true_mask) * 1e12
			y_pred_pos = cosine_score_pos - (1 - pos_true_mask) * 1e12

			# add circle-loss without margin and scale-factor
			joint_neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
			joint_pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
			cpc_loss = tf.reduce_sum(tf.nn.softplus(joint_neg_loss+joint_pos_loss)*loss_mask)/tf.reduce_sum(loss_mask+1e-10)
			loss += cpc_loss
			if not kargs.get('use_tpu'):
				tf.summary.scalar("cpc_loss", cpc_loss)
				tf.summary.scalar("cpc_mask", tf.reduce_mean(tf.reduce_sum(loss_mask, axis=-1)))
			tf.logging.info("****** apply cpc-loss *******")

		if kargs.get('apply_gp', False):
			gp_loss = loss_utils.gradient_penalty_loss(loss, model.get_embedding_table(), 
											epsilon=1.0)
			loss += gp_loss
			tf.logging.info("****** apply gradient penalty *******")

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
			# if model_config.get('label_type', 'single_label') == 'single_label':
			# 	print(logits.get_shape(), "===logits shape===")
			# 	pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			# 	prob = tf.nn.softmax(logits)
			# 	max_prob = tf.reduce_max(prob, axis=-1)
				
			# 	estimator_spec = tf.estimator.EstimatorSpec(
			# 							mode=mode,
			# 							predictions={
			# 										'pred_label':pred_label,
			# 										"max_prob":max_prob
			# 							},
			# 							export_outputs={
			# 								"output":tf.estimator.export.PredictOutput(
			# 											{
			# 												'pred_label':pred_label,
			# 												"max_prob":max_prob
			# 											}
			# 										)
			# 							}
			# 				)
			if model_config.get('label_type', 'single_label') == 'multi_label':
				print("==apply multi_label==")
				prob = tf.nn.sigmoid(logits)
				estimator_spec = tf.estimator.EstimatorSpec(
										mode=mode,
										predictions={
													'pred_label':prob,
													"max_prob":prob
										},
										export_outputs={
											"output":tf.estimator.export.PredictOutput(
														{
															'pred_label':prob,
															"max_prob":prob
														}
													)
										}
							)
			elif model_config.get('label_type', 'single_label') == "single_label":
				print("==apply multi_label==")
				prob = tf.nn.softmax(logits)
				estimator_spec = tf.estimator.EstimatorSpec(
										mode=mode,
										predictions={
													'pred_label':prob,
													"max_prob":prob
										},
										export_outputs={
											"output":tf.estimator.export.PredictOutput(
														{
															'pred_label':prob,
															"max_prob":prob
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
				eval_metric_ops = metric_fn( 
							per_example_loss,
							logits, 
							label_ids)
			
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops)
				return estimator_spec
		else:
			raise NotImplementedError()
	return model_fn

