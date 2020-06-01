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
from utils.textcnn import textcnn_utils, dgcnn_utils
from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics
from task_module import pretrain
from utils.bert import bert_utils
from utils.simclr import simclr_utils
from optimizer import distributed_optimizer as optimizer
from utils.vae import vae_utils
from model_io import model_io_utils

def train_metric(input_ids, predicted_logits, **kargs):
	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = predicted_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	sequence_mask = tf.to_float(tf.not_equal(input_ids[:, 1:], 
													kargs.get('[PAD]', 0)))

	per_example_perplexity = tf.reduce_sum(input_id_logits * sequence_mask, axis=-1) # batch
	per_example_perplexity /= (1e-10+tf.reduce_sum(sequence_mask, axis=-1)) # batch

	perplexity = tf.reduce_mean(tf.exp(per_example_perplexity))

	lm_token_accuracy = tf.equal(
						tf.cast(labels, tf.int32),
						tf.cast(tf.argmax(logits, axis=-1), tf.int32))

	lm_token_accuracy = tf.reduce_sum(tf.cast(lm_token_accuracy, tf.float32) * sequence_mask, axis=-1)
	lm_token_accuracy /= (1e-10+tf.reduce_sum(sequence_mask, axis=-1)) # batch

	return {
		"perplexity": perplexity,
		"token_acc": tf.reduce_mean(lm_token_accuracy)
		}


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

		loss_mask = tf.cast(features["{}_loss_multipiler".format(task_type)], tf.float32)
		
		if kargs.get('merge_mode', 'all') == 'all':
			input_ids = tf.concat([features['input_ids_a'], features['input_ids_b']], axis=0)
			hidden_repres = tf.concat([feature_a, feature_b], axis=0)
			sent_repres = tf.concat([pooled_feature_dict['sent_repres_a'], pooled_feature_dict['sent_repres_b']], axis=0)
			tf.logging.info("****** double batch *******")
		else:
			input_ids = features['input_ids_b']
			hidden_repres = feature_b
			sent_repres = pooled_feature_dict['sent_repres_b']
			tf.logging.info("****** single batch b *******")
		sequence_mask = tf.to_float(tf.not_equal(input_ids, 
											kargs.get('[PAD]', 0)))

		with tf.variable_scope("vae/connect", reuse=tf.AUTO_REUSE):
			with tf.variable_scope("z_mean"):
				z_mean = tf.layers.dense(
							hidden_repres,
							128,
							use_bias=None,
							activation=None,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
				bn_z_mean = vae_utils.mean_normalize_scale(z_mean, 
												is_training, 
												"bn_mean", 
												tau=0.5,
												reuse=tf.AUTO_REUSE,
												**kargs)

			with tf.variable_scope("z_std"):
				z_std = tf.layers.dense(
							hidden_repres,
							128,
							use_bias=True,
							activation=tf.nn.relu,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))	
				bn_z_std = vae_utils.std_normalize_scale(z_std, 
							is_training, 
							"bn_std", 
							tau=0.5,
							reuse=tf.AUTO_REUSE,
							**kargs)

			gaussian_noise = vae_utils.hidden_sampling(bn_z_mean, bn_z_std, **kargs)
			# sent_repres_shape = bert_utils.get_shape_list(sent_repres, expected_rank=[2,3])
			# with tf.variable_scope("vae/projection"):
			# 	gaussian_noise = tf.layers.dense(
			# 				gaussian_noise,
			# 				sent_repres_shape[-1],
			# 				use_bias=None,
			# 				activation=None,
			# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

		# with tf.variable_scope("vae/decoder", reuse=tf.AUTO_REUSE):
		# 	sequence_output = dgcnn_utils.dgcnn(
		# 										sent_repres, 
		# 										sequence_mask,
		# 										num_layers=model_config['cnn_num_layers'], 
		# 										dilation_rates=model_config.get('cnn_dilation_rates', [1,2]),
		# 										strides=model_config.get('cnn_dilation_rates', [1,1]),
		# 										num_filters=model_config.get('cnn_num_filters', [128,128]), 
		# 										kernel_sizes=model_config.get('cnn_filter_sizes', [3,3]), 
		# 										is_training=is_training,
		# 										scope_name="textcnn_encoder/textcnn/forward", 
		# 										reuse=tf.AUTO_REUSE, 
		# 										activation=tf.nn.relu,
		# 										is_casual=model_config['is_casual'],
		# 										padding=model_config.get('padding', 'same')
		# 										)
		# 	sequence_output_logits = model.build_other_output_logits(sequence_output, reuse=tf.AUTO_REUSE)
		# resc_loss = vae_utils.reconstruction_loss(sequence_output_logits, 
		# 										input_ids,
		# 										name="decoder_resc",
		# 										use_tpu=False)
		with tf.variable_scope("vae/bow_resc", reuse=tf.AUTO_REUSE):
			bow_loss, bow_logits = vae_utils.bow_loss(input_ids, gaussian_noise, 
				128, model_config.vocab_size, is_training, 
				bow_loss="term_binary",
				name="vae_bow",
				use_tpu=False)
		
		kl_loss = vae_utils.kl_loss(bn_z_mean, bn_z_std, 
									opt_config.get('num_train_steps', 10000), 
									name="kl_div",
									use_tpu=False,
									kl_anneal="kl_anneal")
		loss = bow_loss + kl_loss
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
		vae_tvars = model_io_fn.get_params("vae", 
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
		# print(tvars)
		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			[assignment_map, 
			initialized_variable_names] = model_io_utils.get_assigment_map_from_checkpoint(
															tvars, 
															init_checkpoint,
															exclude_scope="")
			[assignment_map_vae, 
			initialized_variable_names_vae] = model_io_utils.get_assigment_map_from_checkpoint(
															vae_tvars, 
															init_checkpoint,
															exclude_scope="vae/decoder")
			assignment_map.update(assignment_map_vae)
			initialized_variable_names.update(initialized_variable_names_vae)
			if use_tpu == 0:
				model_io_utils.init_pretrained(assignment_map, initialized_variable_names,
										tvars+vae_tvars, init_checkpoint)
			else:
				tf.logging.info(" initializing parameter from init checkpoint ")
				def tpu_scaffold():
					model_io_utils.init_pretrained(assignment_map, initialized_variable_names,
										tvars+vae_tvars, init_checkpoint)
					return tf.train.Scaffold()
				scaffold_fn = tpu_scaffold

		if mode == tf.estimator.ModeKeys.TRAIN:

			# train_metric_dict = train_metric(input_ids, 
			# 								sequence_output_logits,
			# 									**kargs)
			return_dict = {
					"loss":loss, 
					"tvars":tvars+vae_tvars
				}
			# return_dict["perplexity"] = train_metric_dict['perplexity']
			# return_dict["token_acc"] = train_metric_dict['token_acc']
			return_dict["kl_div"] = kl_loss
			return_dict["kl_bow"] = bow_loss
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


		

				