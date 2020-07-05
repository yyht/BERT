try:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils
	from loss import loss_utils, triplet_loss_utils
except:
	from distributed_single_sentence_classification.model_interface import model_zoo
	from distillation import distillation_utils
	from loss import loss_utils, triplet_loss_utils

import tensorflow as tf
import numpy as np

from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics
from task_module import pretrain
from utils.bert import bert_utils
from utils.simclr import simclr_utils
from optimizer import distributed_optimizer as optimizer

def get_labels_of_similarity(query_input_ids, anchor_query_ids):
	idxs_1 = tf.expand_dims(query_input_ids, axis=1) # batch 1 seq
	idxs_2 = tf.expand_dims(anchor_query_ids, axis=0) # 1 batch seq
	# batch x batch x seq
	labels = tf.cast(tf.not_equal(idxs_1, idxs_2), tf.float32) # not equal:1, equal:0
	equal_num = tf.reduce_sum(labels, axis=-1) # [batch, batch]
	not_equal_label = tf.cast(tf.not_equal(equal_num, 0), tf.float32)
	equal_label = tf.cast(tf.equal(equal_num, 0), tf.float32)
	not_equal_label_shape = bert_utils.get_shape_list(not_equal_label, expected_rank=[2,3])
	not_equal_label *= tf.cast(1 - tf.eye(not_equal_label_shape[0]), tf.float32) 
	return not_equal_label, equal_label

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
		num_task = kargs.get('num_task', 1)
		temp = kargs.get('temp', 0.1)

		print("==task_type==", task_type)

		model_io_fn = model_io.ModelIO(model_io_config)
		label_ids = tf.cast(features["{}_label_ids".format(task_type)], dtype=tf.int32)

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

		shape_list = bert_utils.get_shape_list(pooled_feature_dict['feature_a'], 
												expected_rank=[2])
		batch_size = shape_list[0]

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
		else:
			feature_a = pooled_feature_dict['feature_a']
			feature_b = pooled_feature_dict['feature_b']
			tf.logging.info("****** not apply projection *******")

		# feature_a = tf.nn.l2_normalize(feature_a, axis=-1)
		# feature_b = tf.nn.l2_normalize(feature_b, axis=-1)
		# [batch_size, batch_size]
		if kargs.get("task_seperate_proj", False):
			if task_type == 'xquad' or task_type == 'wsdm':
				# for passage representation
				with tf.variable_scope(scope+"/{}/feature_output_b".format(task_type), reuse=tf.AUTO_REUSE):
					feature_b = tf.layers.dense(
							feature_b,
							128,
							use_bias=True,
							activation=tf.tanh,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
				tf.logging.info("****** apply passage projection *******")
			if task_type == 'afqmc':
				# for anchor representation
				with tf.variable_scope(scope+"/{}/feature_output_a".format(task_type), reuse=tf.AUTO_REUSE):
					feature_a = tf.layers.dense(
							feature_a,
							128,
							use_bias=True,
							activation=tf.tanh,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
				# for successor representation
				with tf.variable_scope(scope+"/{}/feature_output_b".format(task_type), reuse=tf.AUTO_REUSE):
					feature_b = tf.layers.dense(
							feature_b,
							128,
							use_bias=True,
							activation=tf.tanh,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
				tf.logging.info("****** apply cpc anchor, successor projection *******")
			
		if kargs.get("apply_l2_normalize", True):
			feature_a = tf.nn.l2_normalize(feature_a+1e-20, axis=-1)
			feature_b = tf.nn.l2_normalize(feature_b+1e-20, axis=-1)
			tf.logging.info("****** apply normalization *******")

		cosine_score = tf.matmul(feature_a, tf.transpose(feature_b)) # / model_config.get('temperature', 1.0)
		tf.logging.info("****** temperature *******", str(model_config.get('temperature', 1.0)))
		print("==cosine_score shape==", cosine_score.get_shape())
		loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)
		
		if task_type == 'xquad':
			neg_true_mask = tf.cast(triplet_loss_utils._get_anchor_negative_triplet_mask(label_ids), tf.float32)
			pos_true_mask = (1.0 - neg_true_mask) * tf.expand_dims(loss_mask, axis=-1) * tf.expand_dims(loss_mask, axis=0)
			neg_true_mask = neg_true_mask * tf.expand_dims(loss_mask, axis=-1) * tf.expand_dims(loss_mask, axis=0)
		elif task_type in ['wsdm', 'brand_search']:
			pos_label_mask = tf.cast(features["{}_label_ids".format(task_type)], dtype=tf.float32)
			pos_label_mask = tf.expand_dims(pos_label_mask, axis=0)  #* tf.expand_dims(pos_label_mask, axis=0)
			[not_equal_mask, equal_mask] = get_labels_of_similarity(
									features['input_ids_a'], 
									features['input_ids_a'])
			pos_label_mask *= equal_mask
			pos_true_mask = pos_label_mask
			label_mask_ = tf.minimum(tf.reduce_sum(pos_true_mask, axis=-1), 1.0)
			loss_mask *= label_mask_ # remove none-positive part
			score_shape = bert_utils.get_shape_list(cosine_score, expected_rank=[2,3])
			neg_true_mask = tf.ones_like(cosine_score) - pos_true_mask
			pos_true_mask = pos_true_mask * tf.expand_dims(loss_mask, axis=-1) * tf.expand_dims(loss_mask, axis=0)
			neg_true_mask = neg_true_mask * tf.expand_dims(loss_mask, axis=-1) * tf.expand_dims(loss_mask, axis=0)
		elif task_type in ['afqmc']:
			score_shape = bert_utils.get_shape_list(cosine_score, expected_rank=[2,3])
			[not_equal_mask, equal_mask] = get_labels_of_similarity(
									features['input_ids_a'], 
									features['input_ids_b'])
			pos_true_mask = tf.expand_dims(loss_mask, axis=-1) * tf.eye(score_shape[0]) 
			neg_true_mask = not_equal_mask * tf.expand_dims(loss_mask, axis=-1) * tf.expand_dims(loss_mask, axis=0)

		if kargs.get('if_apply_circle_loss', True):
			logits = loss_utils.circle_loss(cosine_score, 
									pos_true_mask, 
									neg_true_mask,
									margin=0.25,
									gamma=64)
			tf.logging.info("****** apply actual-circle loss *******")
		else:
			cosine_score_neg = neg_true_mask * cosine_score
			cosine_score_pos = -pos_true_mask * cosine_score

			y_pred_neg = cosine_score_neg - (1 - neg_true_mask) * 1e12
			y_pred_pos = cosine_score_pos - (1 - pos_true_mask) * 1e12

			# add circle-loss without margin and scale-factor
			joint_neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
			joint_pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
			logits = tf.nn.softplus(joint_neg_loss+joint_pos_loss)

			tf.logging.info("****** apply normal-circle loss *******")

		loss = tf.reduce_sum(logits*loss_mask) / (1e-10+tf.reduce_sum(loss_mask))
		task_loss = loss
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

		if mode == tf.estimator.ModeKeys.TRAIN:
			multi_task_config = kargs.get("multi_task_config", {})
			if multi_task_config.get(task_type, {}).get("lm_augumentation", False):
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
			if multi_task_config.get(task_type, {}).get("lm_augumentation", False):
				print("==apply lm_augumentation==")
				masked_lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
												not_storage_params=not_storage_params)
				tvars.extend(masked_lm_pretrain_tvars)

		try:
			params_size = model_io_fn.count_params(model_config.scope)
			print("==total params==", params_size)
		except:
			print("==not count params==")
		# # print(tvars)
		# if load_pretrained == "yes":
		# 	model_io_fn.load_pretrained(tvars, 
		# 								init_checkpoint,
		# 								exclude_scope=exclude_scope)

		use_tpu = 1 if kargs.get('use_tpu', False) else 0

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu)
		else:
			scaffold_fn = None

		if mode == tf.estimator.ModeKeys.TRAIN:

			# acc = build_accuracy(logits, 
			# 					label_ids, 
			# 					loss_mask,
			# 					loss_type=kargs.get('loss', 'contrastive_loss'))

			return_dict = {
					"loss":loss, 
					"logits":logits,
					"task_num":tf.reduce_sum(loss_mask),
					"{}_pos_num".format(task_type):tf.reduce_sum(pos_true_mask),
					"{}_neg_num".format(task_type):tf.reduce_sum(neg_true_mask),
					"tvars":tvars
				}
			# return_dict["{}_acc".format(task_type)] = acc
			if kargs.get("task_invariant", "no") == "yes":
				return_dict["{}_task_loss".format(task_type)] = masked_task_loss
				task_acc = build_accuracy(task_logits, features["task_id"], loss_mask)
				return_dict["{}_task_acc".format(task_type)] = task_acc
			if multi_task_config.get(task_type, {}).get("lm_augumentation", False):
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


		

				