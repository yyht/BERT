import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

import tensorflow as tf
import numpy as np
from optimizer import optimizer
from optimizer import distributed_optimizer
from model_io import model_io

from task_module import classifier
from task_module import tsa_pretrain
import tensorflow as tf
from metric import tf_metrics

from pretrain_finetuning.token_discriminator import classifier as disc_classifier
from pretrain_finetuning.token_generator import token_generator, random_input_ids_generation
from pretrain_finetuning.token_generator_hmm import hmm_input_ids_generation, ngram_prob
from utils.sampling_utils.glancing_sampling_utils import glance_sample
from task_module import mixup_represt_learning
# from utils.adversarial_utils import adversarial_utils

def train_metric_fn(masked_lm_example_loss, masked_lm_log_probs, 
					masked_lm_ids,
					masked_lm_weights, 
					next_sentence_example_loss,
					next_sentence_log_probs, 
					next_sentence_labels,
					**kargs):
	"""Computes the loss and accuracy of the model."""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
									 [-1, masked_lm_log_probs.shape[-1]])
	masked_lm_predictions = tf.argmax(
		masked_lm_log_probs, axis=-1, output_type=tf.int32)
	masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
	masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
	masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
	masked_lm_weights = tf.cast(masked_lm_weights, tf.float32)

	masked_lm_mask = kargs.get('masked_lm_mask', None)
	if masked_lm_mask is not None:
		masked_lm_mask = tf.reshape(masked_lm_mask, [-1])
		masked_lm_weights *= tf.cast(masked_lm_mask, tf.float32)

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
						**kargs):
	model_config.tsa = 'exp_schedule'
	model_config.num_train_steps = opt_config.num_train_steps
	# opt_config.init_lr /= 2

	ngram_list = kargs.get("ngram", [10, 5, 3])
	mask_prob_list = kargs.get("mask_prob", [0.2, 0.2, 0.2])
	ngram_ratio = kargs.get("ngram_ratio", [7, 1, 1])
	uniform_ratio = kargs.get("uniform_ratio", 0.1)
	tf.logging.info("****** dynamic ngram: %s, mask_prob: %s, mask_prior: %s, uniform_ratio: %s *******", 
			str(ngram_list), str(mask_prob_list), str(ngram_ratio), str(uniform_ratio))	
	tran_prob_list, hmm_tran_prob_list = [], []
	for ngram_sub, mask_prob_sub in zip(ngram_list, mask_prob_list):
		tran_prob, hmm_tran_prob = ngram_prob(ngram_sub, mask_prob_sub)
		tran_prob_list.append(tran_prob)
		hmm_tran_prob_list.append(hmm_tran_prob)
	mask_prior = []
	for ratio in ngram_ratio:
		actual_ratio = (1 - uniform_ratio) / sum(ngram_ratio) * ratio
		mask_prior.append(actual_ratio)
	mask_prior.append(uniform_ratio)
	mask_prior = np.array(mask_prior).astype(np.float32)

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)
		print(features)
		if "pad_mask" in features:
			input_mask = features['pad_mask']
			features['input_mask'] = tf.identity(input_mask)
		if 'input_mask' not in features:
			input_mask = tf.cast(tf.not_equal(features['input_ids_{}'.format(target)], 
																			kargs.get('[PAD]', 0)), tf.int32)

			if target:
				features['input_mask_{}'.format(target)] = input_mask
			else:
				features['input_mask'] = tf.identity(input_mask)
		if 'segment_ids' not in features:
			segment_ids = tf.zeros_like(input_mask)
			if target:
				features['segment_ids_{}'.format(target)] = segment_ids
			else:
				features['segment_ids'] = tf.identity(segment_ids)

		if target:
			features['input_ori_ids'] = features['input_ids_{}'.format(target)]
			features['input_mask'] = features['input_mask_{}'.format(target)]
			features['segment_ids'] = features['segment_ids_{}'.format(target)]
			features['input_ids'] = features['input_ids_{}'.format(target)]

		if 'input_ori_ids' in features:
			input_ori_ids = features['input_ori_ids']
			tf.logging.info("***** original input ori ids *****")
		elif 'origin_input' in features:
			input_ori_ids = features['origin_input']
			features['input_ori_ids'] = features['origin_input']
			tf.logging.info("***** origin_input *****")
		else:
			input_ori_ids = None
			tf.logging.info("***** no origin_input *****")
		if 'masked_input' in features:
			features['input_ids'] = tf.identity(features['masked_input'])
			model_config.corrupted = False

		if mode == tf.estimator.ModeKeys.TRAIN:
			is_training = True
			if input_ori_ids is not None and model_config.get("corrupted", True):
				# [output_ids, 
				# sampled_binary_mask] = random_input_ids_generation(
				# 							model_config, 
				# 							input_ori_ids,
				# 							features['input_mask'],
				# 							**kargs)

				[output_ids, 
				sampled_binary_mask] = hmm_input_ids_generation(model_config,
										features['input_ori_ids'],
										features['input_mask'],
										[tf.cast(tf.constant(hmm_tran_prob), tf.float32) for hmm_tran_prob in hmm_tran_prob_list],
										mask_probability=0.2,
										replace_probability=0.1,
										original_probability=0.1,
										mask_prior=tf.cast(tf.constant(mask_prior), tf.float32),
										**kargs)

				features['input_ids'] = output_ids
				tf.logging.info("***** Running random sample input generation *****")
			else:
				sampled_binary_mask = features['input_mask']
				tf.logging.info("***** Running original sample input generation *****")
		else:
			sampled_binary_mask = features['input_mask']
			tf.logging.info("***** Running original sample input generation *****")
			is_training = False

		model_features = {}
		for key in features:
			model_features[key] = tf.identity(features[key])

		if model_config.get("model_type", "bert") == "funnelbert":
			"""
			funnel-bert needs opposite pad-mask as input
			"""
			model_features['input_mask'] = (1.0 - tf.cast(model_features['input_mask'], dtype=tf.float32))
			tf.logging.info("***** funnelbert needs reverse input-mask *****")
			cls_token_type = model_config.get('seg_id_cls', 2) * tf.ones_like(model_features['segment_ids'][:, 0:1])
			model_features['segment_ids'] = tf.concat([cls_token_type, model_features['segment_ids'][:, 1:]], axis=1)
			model_features['normal_input_mask'] = tf.identity(model_features['input_mask'])

			n_block = len(model_config.get('block_size', "4").split("_"))
			if n_block > 1:
				return_type = "decoder"
				if_use_decoder = 'use_decoder'
				tf.logging.info("***** apply decoder reconstruction *****")
			else:
				return_type = "encoder"
				if_use_decoder = 'none'
				tf.logging.info("***** apply encoder reconstruction *****")
			if n_block > 1:
				if model_config.get("denoise_mode", "autoencoding") == "autoencoding":
					model_features['input_ids'] = tf.identity(input_ori_ids)
					tf.logging.info("***** apply auto-encoding reconstruction *****")
				elif model_config.get("denoise_mode", "autoencoding") == "denoise":
					tf.logging.info("***** apply denoise reconstruction *****")
				elif model_config.get("denoise_mode", "autoencoding") == "text_infilling":
					if 'infilled_input' in features:
						model_features['input_ids'] = tf.identity(features['infilled_input'])
						tf.logging.info("***** apply etxt text_infilling reconstruction *****")
						features['input_mask'] = tf.identity(features['infilling_pad_mask'])
						tf.logging.info("***** masked input-ids with infilling mask and rec-mask *****")
					tf.logging.info("***** apply denoise reconstruction *****")
			else:
				tf.logging.info("***** apply mlm-denoise *****")
		else:
			return_type = 'encoder'
			if_use_decoder = 'none'
			tf.logging.info("***** apply standard bert-mlm *****")

		model = model_api(model_config, model_features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							if_use_decoder=if_use_decoder,
							**kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		if 'label' in model_features:
			(nsp_loss, 
			nsp_per_example_loss, 
			nsp_log_prob) = pretrain.get_next_sentence_output(model_config,
											model.get_pooled_output(),
											features['label'],
											reuse=tf.AUTO_REUSE)

		# masked_lm_positions = features["masked_lm_positions"]
		# masked_lm_ids = features["masked_lm_ids"]
		# masked_lm_weights = features["masked_lm_weights"]

		if model_config.model_type == 'bert':
			masked_lm_fn = pretrain.get_masked_lm_output
			seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
			print("==apply bert masked lm==")
		elif model_config.model_type == 'albert':
			masked_lm_fn = pretrain_albert.get_masked_lm_output
			seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
			print("==apply albert masked lm==")
		elif model_config.model_type == 'funnelbert':
			masked_lm_fn = pretrain.get_masked_lm_output
			seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
			print("==apply funnelbert masked lm==")
		else:
			masked_lm_fn = pretrain.get_masked_lm_output
			seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
			print("==apply bert masked lm==")

		if model_config.get("model_type", "bert") == "funnelbert":
			if n_block > 1 and model_config.get('pretrain_loss', "ae") == "ae":
				seq_masked_lm_fn = pretrain.denoise_autoencoder
				discriminator_mode = model_config.get('discriminator_mode', "normal_ce_loss")
				loss_converage = model_config.get("loss_converage", 'global')
				tf.logging.info("***** discriminator_mode: %s *****"%(discriminator_mode))
				tf.logging.info("***** loss_converage: %s *****"%(loss_converage))
				tf.logging.info(seq_masked_lm_fn)
				model_config.corrupted = True
				tf.logging.info("*** apply reconstruction ***")
				if loss_converage in ['global']:
					sampled_binary_mask = tf.identity(features['input_mask'])
					tf.logging.info("***** loss_converage: %s ***** with input-mask"%(loss_converage))
				elif loss_converage in ['local']:
					sampled_binary_mask = tf.reduce_sum(features['target_mapping'], axis=1)
					tf.logging.info("***** loss_converage: %s ***** with target-mapping mask"%(loss_converage))
			else:
				discriminator_mode = model_config.get('discriminator_mode', "ce_loss")
				loss_converage = model_config.get("loss_converage", 'global')
				tf.logging.info(seq_masked_lm_fn)
		else:
			discriminator_mode = "ce_loss"
			loss_converage = model_config.get("loss_converage", 'global')
			tf.logging.info(seq_masked_lm_fn)
			tf.logging.info(masked_lm_fn)

		if input_ori_ids is not None and model_config.get("corrupted", True) or model_config.get("all_tokens", False):
			(masked_lm_loss,
			masked_lm_example_loss, 
			masked_lm_log_probs,
			masked_lm_mask) = seq_masked_lm_fn(model_config, 
										model.get_sequence_output(output_type=return_type), 
										model.get_embedding_table(),
										features['input_mask'], 
										features['input_ori_ids'], 
										features['input_ids'],
										sampled_binary_mask,
										reuse=tf.AUTO_REUSE,
										embedding_projection=model.get_embedding_projection_table(),
										pretrain_loss_type="normal",
										discriminator_mode=discriminator_mode,
										loss_converage=loss_converage)
			masked_lm_ids = input_ori_ids
			tf.logging.info("*** apply sequential mlm loss ***")
		else:
			masked_lm_positions = features["masked_lm_positions"]
			masked_lm_ids = features["masked_lm_ids"]
			masked_lm_weights = features["masked_lm_weights"]

			(pre_masked_lm_loss,
			pre_masked_lm_example_loss, 
			pre_masked_lm_log_probs,
			pre_masked_lm_mask) = masked_lm_fn(
											model_config, 
											model.get_sequence_output(output_type=return_type), 
											model.get_embedding_table(),
											masked_lm_positions, 
											masked_lm_ids, 
											masked_lm_weights,
											reuse=tf.AUTO_REUSE,
											embedding_projection=model.get_embedding_projection_table(),
											pretrain_loss_type="normal",
											discriminator_mode=discriminator_mode,
											loss_converage=loss_converage)
			tf.logging.info("*** apply bert-like mlm loss ***")

			# glancing_training
			if kargs.get("glancing_training", "none") == "none":
				tf.logging.info("*** no need glancing_training ***")
				masked_lm_loss = tf.identity(pre_masked_lm_loss)
			else:
				tf.logging.info("*** glancing_training ***")
				[
					output_ids, 
					none_glanced_masked_lm_positions,
					none_glanced_masked_lm_ids,
					none_glanced_lm_weights
				] = glance_sample(pre_masked_lm_log_probs,
							features["masked_lm_positions"],
							features["masked_lm_ids"],
							features["masked_lm_weights"],
							model_features['input_ids'],
							model_features['input_ori_ids'],
							features['input_mask'],
							opt_config.get('num_train_steps', 100000),
							model_config.vocab_size,
							use_tpu=kargs.get('use_tpu', False))

				glance_model_features = {}
				for key in model_features:
					glance_model_features[key] = tf.identity(model_features[key])
				glance_model_features['input_ids'] = tf.identity(output_ids)
				glance_model = model_api(model_config, glance_model_features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							if_use_decoder=if_use_decoder,
							**kargs)
				(glance_masked_lm_loss,
				glance_masked_lm_example_loss, 
				glance_masked_lm_log_probs,
				glance_masked_lm_mask) = masked_lm_fn(
												model_config, 
												glance_model.get_sequence_output(output_type=return_type), 
												glance_model.get_embedding_table(),
												none_glanced_masked_lm_positions,
												none_glanced_masked_lm_ids,
												none_glanced_lm_weights,
												reuse=tf.AUTO_REUSE,
												embedding_projection=model.get_embedding_projection_table(),
												pretrain_loss_type="normal",
												discriminator_mode=discriminator_mode,
												loss_converage=loss_converage)
				masked_lm_loss = tf.identity(glance_masked_lm_loss)
		print(model_config.lm_ratio, '==mlm lm_ratio==')
		loss = model_config.lm_ratio * masked_lm_loss #+ 0.0 * nsp_loss
		if 'label' in model_features:
			loss += nsp_loss

		if kargs.get("apply_mixup", "none") == 'mixup':

			mixup_features = {}
			for key in features:
				mixup_features[key] = tf.identity(model_features[key])
			mixup_features['input_ids'] = tf.identity(features['origin_input'])
			mixup_model = model_api(model_config, mixup_features, labels,
								mode, target, reuse=tf.AUTO_REUSE,
								if_use_decoder=if_use_decoder,
								**kargs)

			tpu_context = params['context'] if 'context' in params else None
			simclr_config = {
				"proj_out_dim":model_config.hidden_size * 4,
				"proj_head_mode":"nonlinear",
				"num_proj_layers":2
			}
			contrast_loss = mixup_represt_learning.mixup_dsal_plus(
					config=simclr_config,
					 hidden=mixup_model.get_sequence_output(),
			        input_mask=mixup_features['input_mask'],
			        temperature=0.1,
			        hidden_norm=True,
			        masked_repres=None,
			        is_training=is_training,
			        beta=0.5,
			        use_bn=True,
			        tpu_context=tpu_context,
			        weights=1.0)
			loss += contrast_loss
		# if kargs.get("apply_vat", False):

		# 	adv_features = {}
		# 	for key in model_features:
		# 		adv_features[key] = tf.identity(model_features[key])

		# 	unk_mask = tf.cast(tf.math.equal(adv_features['input_ids'], 100), tf.float32) # not replace unk
		# 	cls_mask =  tf.cast(tf.math.equal(adv_features['input_ids'], 101), tf.float32) # not replace cls
		# 	sep_mask = tf.cast(tf.math.equal(adv_features['input_ids'], 102), tf.float32) # not replace sep
		# 	mask_mask = tf.cast(tf.math.equal(adv_features['input_ids'], 103), tf.float32) # not replace sep
		# 	none_replace_mask =  unk_mask + cls_mask + sep_mask + mask_mask
		# 	noise_mask = tf.cast(features['input_mask'], tf.float32) * (1-none_replace_mask)
		# 	noise_mask = tf.expand_dims(noise_mask, axis=-1)

		# 	vat_loss = adversarial_utils.adversarial_loss(
		# 					model_config,
		# 					model_api, 
		# 					adv_features, 
		# 					labels,
		# 					adv_masked_lm_log_probs,
		# 					mode,
		# 					target,
		# 					embedding_table=model.get_embedding_table(),
		# 					noise_mask=noise_mask,
		# 					embedding_seq_output=model.get_embedding_output(),
		# 					sampled_binary_mask=sampled_binary_mask,
		# 					num_power_iterations=1,
		# 					noise_var=1e-5,
		# 					step_size=1e-3,
		# 					noise_gamma=1e-5,
		# 					is_training=is_training,
		# 					pretrain_loss_type='normal',
		# 					project_norm_type="inf",
		# 					vat_type="alum",
		# 					adv_type="embedding_seq_output",
		# 					stop_gradient=False,
		# 					kl_inclusive=False,
		# 					emb_adv_pos="emb_adv_post", # emb_adv_post
		# 					**kargs)

		# 	loss += kargs.get("vat_ratio", 1.0) * vat_loss
		# 	tf.logging.info("***** apply vat loss:%s *****" % (str(kargs.get("vat_ratio", 1))))
		
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

		pretrained_tvars.extend(lm_pretrain_tvars)
		if 'label' in model_features:
			nsp_pretrain_tvars = model_io_fn.get_params("cls/seq_relationship", 
									not_storage_params=not_storage_params)
			pretrained_tvars.extend(nsp_pretrain_tvars)

		if kargs.get("unigram_disc", False):
			[output_ids, 
			sampled_binary_mask] = hmm_input_ids_generation(model_config,
										features['input_ori_ids'],
										features['input_mask'],
										[tf.cast(tf.constant(hmm_tran_prob), tf.float32) for hmm_tran_prob in hmm_tran_prob_list],
										mask_probability=0.2,
										replace_probability=1.0,
										original_probability=0.0,
										mask_prior=tf.cast(tf.constant(mask_prior), tf.float32),
										**kargs)
			tf.logging.info("***** apply random sampling *****")
			features['input_ids'] = output_ids

			model = model_api(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)

			with tf.variable_scope('cls/discriminator_predictions', reuse=tf.AUTO_REUSE):
				(disc_loss, 
				logits, 
				per_example_loss) = disc_classifier(model_config, 
										model.get_sequence_output(),
										features['input_ori_ids'],
										features['input_ids'],
										features['input_mask'],
										2,
										dropout_prob,
										use_tpu=kargs.get('use_tpu', False),
										sampled_binary_mask=sampled_binary_mask)
			loss += 50.0*disc_loss
			disc_pretrain_tvars = model_io_fn.get_params("cls/discriminator_predictions", 
										not_storage_params=not_storage_params)
			pretrained_tvars.extend(disc_pretrain_tvars)
		
		# if load_pretrained == "yes":
		# 	scaffold_fn = model_io_fn.load_pretrained(pretrained_tvars, 
		# 									init_checkpoint,
		# 									exclude_scope=exclude_scope,
		# 									use_tpu=1)
		# else:
		# 	scaffold_fn = None

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu)
		else:
			scaffold_fn = None

		if mode == tf.estimator.ModeKeys.TRAIN:
						
			# optimizer_fn = optimizer.Optimizer(opt_config)

			if kargs.get('use_tpu', False):
				optimizer_fn = optimizer.Optimizer(opt_config)
				use_tpu = 1
				tf.logging.info("***** using tpu with tpu-captiable optimizer *****")
			else:
				optimizer_fn = distributed_optimizer.Optimizer(opt_config)
				use_tpu = 0
				tf.logging.info("***** using gpu with gpu-captiable optimizer *****")
						
			tvars = pretrained_tvars
			model_io_fn.print_params(tvars, string=", trainable params")
			
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer_fn.get_train_op(loss, tvars,
								opt_config.init_lr, 
								opt_config.num_train_steps,
								use_tpu=opt_config.use_tpu)

			#	train_metric_dict = train_metric_fn(
			#			masked_lm_example_loss, masked_lm_log_probs, 
			#			masked_lm_ids,
			#			masked_lm_mask, 
			#			nsp_per_example_loss,
			#			nsp_log_prob, 
			#			features['next_sentence_labels'],
			#			masked_lm_mask=masked_lm_mask
			#		)

				# for key in train_metric_dict:
				# 	tf.summary.scalar(key, train_metric_dict[key])
				# tf.summary.scalar('learning_rate', optimizer_fn.learning_rate)

				# estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
				# 				mode=mode,
				# 				loss=loss,
				# 				train_op=train_op,
				# 				scaffold_fn=scaffold_fn)

			if kargs.get('use_tpu', False):
				estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
								mode=mode,
								loss=loss,
								train_op=train_op,
								scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(
								mode=mode, 
								loss=loss, 
								train_op=train_op)

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

			def metric_fn_v1(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
					masked_lm_weights):
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

				return {
					"masked_lm_accuracy": masked_lm_accuracy,
					"masked_lm_loss": masked_lm_mean_loss
					}

			eval_metrics = (metric_fn_v1, [
			  masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
			  masked_lm_mask
			])

			estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
						  mode=mode,
						  loss=loss,
						  eval_metrics=eval_metrics,
						  scaffold_fn=scaffold_fn)

			return estimator_spec
		else:
			raise NotImplementedError()

	return model_fn
