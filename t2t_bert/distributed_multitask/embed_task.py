try:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils
	from loss import loss_utils
except:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils
	from loss import loss_utils

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics
from task_module import pretrain
from utils.bert import bert_utils
from optimizer import distributed_optimizer as optimizer
from utils.simclr import simclr_utils

def build_accuracy(logits, labels, mask, loss_type):
	mask = tf.cast(mask, tf.float32)
	if loss_type == 'contrastive_loss':
		temp_sim = tf.subtract(tf.ones_like(logits), tf.rint(logits), name="temp_sim") #auto threshold 0.5
		correct = tf.equal(
							tf.cast(temp_sim, tf.float32),
							tf.cast(labels, tf.float32)
		)
		accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)*mask)/(1e-10+tf.reduce_sum(mask))
	elif loss_type == 'exponent_neg_manhattan_distance_mse':
		temp_sim = tf.rint(logits)
		correct = tf.equal(
							tf.cast(temp_sim, tf.float32),
							tf.cast(labels, tf.float32)
		)
		accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)*mask)/(1e-10+tf.reduce_sum(mask))
	return accuracy

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

		task_type = kargs.get("task_type", "cls")

		label_ids = tf.cast(features["{}_label_ids".format(task_type)], tf.float32)
		if task_type in ['mnli', 'cmnli']:
			loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)
			nerual_label = tf.not_equal(
							label_ids,
							tf.zeros_like(label_ids)
			)

			pos_label =  tf.equal(
							label_ids,
							tf.ones_like(label_ids)
			)

			neg_label =  tf.not_equal(
							label_ids,
							2*tf.ones_like(label_ids)
			)

			loss_mask *= tf.cast(nerual_label, dtype=tf.float32) # make neural label
			label_ids *= tf.cast(neg_label, dtype=tf.float32)

		else:
			loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)

		num_task = kargs.get('num_task', 1)

		model_io_fn = model_io.ModelIO(model_io_config)

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

		if kargs.get("get_pooled_output", "pooled_output") == "pooled_output":
			pooled_feature = model.get_pooled_output()
		elif kargs.get("get_pooled_output", "task_output") == "task_output":
			pooled_feature_dict = model.get_task_output()
			pooled_feature = pooled_feature_dict['pooled_feature']

		if kargs.get('apply_head_proj', False):
			with tf.variable_scope(scope+"/head_proj", reuse=tf.AUTO_REUSE):
				feature_a = simclr_utils.projection_head(pooled_feature_dict['feature_a'], 
										is_training, 
										head_proj_dim=128,
										num_nlh_layers=1,
										head_proj_mode='nonlinear',
										name='head_contrastive')
				pooled_feature_dict['feature_a'] = feature_a

			with tf.variable_scope(scope+"/head_proj", reuse=tf.AUTO_REUSE):
				feature_b = simclr_utils.projection_head(pooled_feature_dict['feature_b'], 
										is_training, 
										head_proj_dim=128,
										num_nlh_layers=1,
										head_proj_mode='nonlinear',
										name='head_contrastive')
				pooled_feature_dict['feature_b'] = feature_b
			tf.logging.info("****** apply contrastive feature projection *******")		

		loss = tf.constant(0.0)

		params_size = model_io_fn.count_params(model_config.scope)
		print("==total encoder params==", params_size)

		if kargs.get("feature_distillation", True):
			universal_feature_a = features.get("input_ids_a_features", None)
			universal_feature_b = features.get("input_ids_b_features", None)
			
			if universal_feature_a is None or universal_feature_b is None:
				tf.logging.info("****** not apply feature distillation *******")
				feature_loss = tf.constant(0.0)
			else:
				feature_a = pooled_feature_dict['feature_a']
				feature_a_shape = bert_utils.get_shape_list(feature_a, expected_rank=[2,3])
				pretrain_feature_a_shape = bert_utils.get_shape_list(universal_feature_a, expected_rank=[2,3])
				if feature_a_shape[-1] != pretrain_feature_a_shape[-1]:
					with tf.variable_scope(scope+"/feature_proj", reuse=tf.AUTO_REUSE):
						proj_feature_a = tf.layers.dense(feature_a, pretrain_feature_a_shape[-1])
					# with tf.variable_scope(scope+"/feature_rec", reuse=tf.AUTO_REUSE):
					# 	proj_feature_a_rec = tf.layers.dense(proj_feature_a, feature_a_shape[-1])
					# loss += tf.reduce_mean(tf.reduce_sum(tf.square(proj_feature_a_rec-feature_a), axis=-1))/float(num_task)
					tf.logging.info("****** apply auto-encoder for feature compression *******")
				else:
					proj_feature_a = feature_a
				feature_a_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(tf.pow(proj_feature_a, 2), axis=-1, keepdims=True))+1e-20)
				proj_feature_a /= feature_a_norm

				feature_b = pooled_feature_dict['feature_b'] 
				if feature_a_shape[-1] != pretrain_feature_a_shape[-1]:
					with tf.variable_scope(scope+"/feature_proj", reuse=tf.AUTO_REUSE):
						proj_feature_b = tf.layers.dense(feature_b, pretrain_feature_a_shape[-1])
					# with tf.variable_scope(scope+"/feature_rec", reuse=tf.AUTO_REUSE):
					# 	proj_feature_b_rec = tf.layers.dense(proj_feature_b, feature_a_shape[-1])
					# loss += tf.reduce_mean(tf.reduce_sum(tf.square(proj_feature_b_rec-feature_b), axis=-1))/float(num_task)
					tf.logging.info("****** apply auto-encoder for feature compression *******")
				else:
					proj_feature_b = feature_b

				feature_b_norm = tf.stop_gradient(tf.sqrt(tf.reduce_sum(tf.pow(proj_feature_b, 2), axis=-1, keepdims=True))+1e-20)
				proj_feature_b /= feature_b_norm

				feature_a_distillation = tf.reduce_mean(tf.square(universal_feature_a-proj_feature_a), axis=-1)
				feature_b_distillation = tf.reduce_mean(tf.square(universal_feature_b-proj_feature_b), axis=-1)

				feature_loss = tf.reduce_mean((feature_a_distillation + feature_b_distillation)/2.0)/float(num_task)
				loss += feature_loss
				tf.logging.info("****** apply prertained feature distillation *******")

		if kargs.get("embedding_distillation", True):
			word_embed = model.emb_mat
			random_embed_shape = bert_utils.get_shape_list(word_embed, expected_rank=[2,3])
			print("==random_embed_shape==", random_embed_shape)
			pretrained_embed = kargs.get('pretrained_embed', None)
			if pretrained_embed is None:
				tf.logging.info("****** not apply prertained feature distillation *******")
				embed_loss = tf.constant(0.0)
			else:
				pretrain_embed_shape = bert_utils.get_shape_list(pretrained_embed, expected_rank=[2,3])
				print("==pretrain_embed_shape==", pretrain_embed_shape)
				if random_embed_shape[-1] != pretrain_embed_shape[-1]:
					with tf.variable_scope(scope+"/embedding_proj", reuse=tf.AUTO_REUSE):
						proj_embed = tf.layers.dense(word_embed, pretrain_embed_shape[-1])
				else:
					proj_embed = word_embed
				
				embed_loss = tf.reduce_mean(tf.reduce_mean(tf.square(proj_embed-pretrained_embed), axis=-1))/float(num_task)
				loss += embed_loss
				tf.logging.info("****** apply prertained feature distillation *******")

		if kargs.get('loss', 'contrastive_loss') == 'contrastive_loss':

			# feature_a = tf.nn.l2_normalize(pooled_feature_dict['feature_a'], axis=-1)
			# feature_b = tf.nn.l2_normalize(pooled_feature_dict['feature_b'], axis=-1)

			# feature_a = pooled_feature_dict['feature_a']
			# feature_b = pooled_feature_dict['feature_b']

			per_example_loss, logits = loss_utils.contrastive_loss(label_ids, 
									pooled_feature_dict['feature_a'],
									pooled_feature_dict['feature_b'],
									kargs.get('margin', 1.0))
			tf.logging.info("****** contrastive_loss *******")
		elif kargs.get('loss', 'contrastive_loss') == 'exponent_neg_manhattan_distance_mse':
			# feature_a = tf.nn.l2_normalize(pooled_feature_dict['feature_a'], axis=-1)
			# feature_b = tf.nn.l2_normalize(pooled_feature_dict['feature_b'], axis=-1)

			per_example_loss, logits = loss_utils.exponent_neg_manhattan_distance(label_ids, 
									pooled_feature_dict['feature_a'],
									pooled_feature_dict['feature_b'],
									'mse')
			tf.logging.info("****** exponent_neg_manhattan_distance_mse *******")
		else:

			per_example_loss, logits = loss_utils.contrastive_loss(label_ids, 
									pooled_feature_dict['feature_a'],
									pooled_feature_dict['feature_b'],
									kargs.get('margin', 1.0))
			tf.logging.info("****** contrastive_loss *******")
		# loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)

		masked_per_example_loss = per_example_loss * loss_mask
		task_loss = tf.reduce_sum(masked_per_example_loss) / (1e-10+tf.reduce_sum(loss_mask))
		loss += task_loss

		# with tf.variable_scope(scope+"/{}/classifier".format(task_type), reuse=task_layer_reuse):
			
		# 	feature_a = pooled_feature_dict['feature_a']
		# 	feature_b = pooled_feature_dict['feature_a']

		# 	logtis_feature = tf.concat([feature_a, feature_b], axis=-1)

		# 	(_, 
		# 		cls_per_example_loss, 
		# 		cls_logits) = classifier.classifier(model_config,
		# 									logtis_feature,
		# 									num_labels,
		# 									label_ids,
		# 									dropout_prob)

		# loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)
		# masked_per_example_loss = cls_per_example_loss * loss_mask
		# task_loss = tf.reduce_sum(masked_per_example_loss) / (1e-10+tf.reduce_sum(loss_mask))
		# loss += task_loss

		if mode == tf.estimator.ModeKeys.TRAIN:
			multi_task_config = kargs.get("multi_task_config", {})
			if multi_task_config[task_type].get("lm_augumentation", False):
				print("==apply lm_augumentation==")
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

				masked_lm_loss_mask = tf.expand_dims(loss_mask, -1) * tf.ones((1, multi_task_config[task_type]["max_predictions_per_seq"]))
				masked_lm_loss_mask = tf.reshape(masked_lm_loss_mask, (-1, ))

				masked_lm_label_weights = tf.reshape(masked_lm_weights, [-1])
				masked_lm_loss_mask *= tf.cast(masked_lm_label_weights, tf.float32)

				masked_lm_example_loss *= masked_lm_loss_mask# multiply task_mask
				masked_lm_loss = tf.reduce_sum(masked_lm_example_loss) / (1e-10+tf.reduce_sum(masked_lm_loss_mask))
				loss += multi_task_config[task_type]["masked_lm_loss_ratio"]*masked_lm_loss

				masked_lm_label_ids = tf.reshape(masked_lm_ids, [-1])
				
				print(masked_lm_log_probs.get_shape(), "===masked lm log probs===")
				print(masked_lm_label_ids.get_shape(), "===masked lm ids===")
				print(masked_lm_label_weights.get_shape(), "===masked lm mask===")

				lm_acc = build_accuracy(masked_lm_log_probs, masked_lm_label_ids, masked_lm_loss_mask)

		if kargs.get("task_invariant", "no") == "yes":
			print("==apply task adversarial training==")
			with tf.variable_scope(scope+"/dann_task_invariant", reuse=model_reuse):
				(_, 
				task_example_loss, 
				task_logits)  = distillation_utils.feature_distillation(model.get_pooled_output(), 
														1.0, 
														features["task_id"], 
														kargs.get("num_task", 7),
														dropout_prob, 
														True)
				masked_task_example_loss = loss_mask * task_example_loss
				masked_task_loss = tf.reduce_sum(masked_task_example_loss) / (1e-10+tf.reduce_sum(loss_mask))
				loss += kargs.get("task_adversarial", 1e-2) * masked_task_loss

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		if mode == tf.estimator.ModeKeys.TRAIN:
			multi_task_config = kargs.get("multi_task_config", {})
			if multi_task_config[task_type].get("lm_augumentation", False):
				print("==apply lm_augumentation==")
				masked_lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
												not_storage_params=not_storage_params)
				tvars.extend(masked_lm_pretrain_tvars)

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

			acc = build_accuracy(logits, 
								label_ids, 
								loss_mask,
								loss_type=kargs.get('loss', 'contrastive_loss'))

			return_dict = {
					"loss":loss, 
					"logits":logits,
					"task_num":tf.reduce_sum(loss_mask),
					"tvars":tvars,
					"positive_label":tf.reduce_sum(label_ids*loss_mask)
				}
			return_dict["{}_acc".format(task_type)] = acc
			if kargs.get("task_invariant", "no") == "yes":
				return_dict["{}_task_loss".format(task_type)] = masked_task_loss
				task_acc = build_accuracy(task_logits, features["task_id"], loss_mask)
				return_dict["{}_task_acc".format(task_type)] = task_acc
			if multi_task_config[task_type].get("lm_augumentation", False):
				return_dict["{}_masked_lm_loss".format(task_type)] = masked_lm_loss
				return_dict["{}_masked_lm_acc".format(task_type)] = lm_acc
			if kargs.get("embedding_distillation", True):
				return_dict["embed_loss"] = embed_loss*float(num_task)
			else:
				return_dict["embed_loss"] = task_loss
			if kargs.get("feature_distillation", True):
				return_dict["feature_loss"] = feature_loss*float(num_task)
			else:
				return_dict["feature_loss"] = task_loss
			return_dict["task_loss"] = task_loss
			return return_dict
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


		

				