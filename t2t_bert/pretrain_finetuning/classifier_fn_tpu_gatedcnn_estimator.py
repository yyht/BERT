import tensorflow as tf
import numpy as np

from optimizer import distributed_optimizer

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

import tensorflow as tf
import numpy as np
from optimizer import optimizer
from model_io import model_io
from utils.bert import bert_seq_utils, bert_seq_sample_utils
from task_module import classifier
from task_module import tsa_pretrain
from pretrain_finetuning.token_discriminator import classifier as disc_classifier
import tensorflow as tf
from metric import tf_metrics

from pretrain_finetuning.token_generator import token_generator, random_input_ids_generation
from pretrain_finetuning.token_generator_hmm import hmm_input_ids_generation, ngram_prob


def train_metric(input_ids, predicted_logits, features, **kargs):
	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = predicted_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	if kargs.get('mask_type', 'left2right') == 'left2right':
		tf.logging.info("***** using left2right mask and loss *****")
		sequence_mask = tf.to_float(tf.not_equal(features['input_ori_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))
	elif kargs.get('mask_type', 'left2right') == 'seq2seq':
		tf.logging.info("***** using seq2seq mask and loss *****")
		sequence_mask = tf.to_float(features['segment_ids'][:, 1:])
		if not kargs.get('use_tpu', False):
			tf.summary.scalar("loss mask", tf.reduce_mean(sequence_mask))

	# sequence_mask = tf.to_float(tf.not_equal(labels, 
	# 							kargs.get('[PAD]', 0)))

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

def eval_metric(input_ids, predicted_logits, features, **kargs):
	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = predicted_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	# sequence_mask = tf.to_float(tf.not_equal(labels, 
	# 							kargs.get('[PAD]', 0)))

	if kargs.get('mask_type', 'left2right') == 'left2right':
		tf.logging.info("***** using left2right mask and loss *****")
		sequence_mask = tf.to_float(tf.not_equal(features['input_ori_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))
	elif kargs.get('mask_type', 'left2right') == 'seq2seq':
		tf.logging.info("***** using seq2seq mask and loss *****")
		sequence_mask = tf.to_float(features['segment_ids'][:, 1:])
		if not kargs.get('use_tpu', False):
			tf.summary.scalar("loss mask", tf.reduce_mean(sequence_mask))

	per_example_perplexity = tf.reduce_sum(input_id_logits * sequence_mask, axis=-1) # batch
	per_example_perplexity /= (1e-10+tf.reduce_sum(sequence_mask, axis=-1)) # batch

	perplexity = tf.exp(per_example_perplexity)

	ppl_avg = tf.metrics.mean(values=perplexity)
	lm_token_accuracy = tf.metrics.accuracy(
					labels=tf.cast(labels, tf.int32), 
					predictions=tf.cast(tf.argmax(logits, axis=-1), tf.int32),
					weights=sequence_mask)

	return {
		"perplexity":ppl_avg,
		"token_acc":lm_token_accuracy
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

	# ngram_list = kargs.get("ngram", [3])
	# mask_prob_list = kargs.get("mask_prob", [0.1])
	# ngram_ratio = kargs.get("ngram_ratio", [8])
	# uniform_ratio = kargs.get("uniform_ratio", 0.1)

	ngram_list = kargs.get("ngram", [10, 3])
	mask_prob_list = kargs.get("mask_prob", [0.2, 0.2])
	ngram_ratio = kargs.get("ngram_ratio", [8, 1])
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

		if target:
			features['input_ori_ids'] = features['input_ids_{}'.format(target)]
			features['input_ids'] = features['input_ids_{}'.format(target)]

		casual_flag = model_config.get('is_casual', True)
		tf.logging.info("***** is casual flag *****", str(casual_flag))

		if 'input_ori_ids' in features:
			input_ori_ids = features['input_ori_ids']
			tf.logging.info("***** original input ori ids *****")
		elif 'origin_input' in features:
			input_ori_ids = features['origin_input']
			features['input_ori_ids'] = tf.identity(features['origin_input'])
			tf.logging.info("***** origin_input *****")
		else:
			input_ori_ids = features['input_ids']
			features['input_ori_ids'] = tf.identity(input_ori_ids)
			tf.logging.info("***** no origin_input *****")

		seq_features = {}
		for key in features:
			seq_features[key] = tf.identity(features[key])
		tf.logging.info(seq_features)

		if input_ori_ids is not None and 'input_mask' not in features:
			sequence_mask = tf.cast(tf.not_equal(input_ori_ids, 
												kargs.get('[PAD]', 0)), 
												tf.int32)
			features['input_mask'] = sequence_mask
			tf.logging.info("***** none-casual and not provided input-mask *****")

		if 'masked_input' in features and not casual_flag:
			seq_features['input_ids'] = features['masked_input']
			model_config.corrupted = False
			tf.logging.info("***** none-casual-mask with masked-input *****")
		if casual_flag :
			model_config.corrupted = False
			tf.logging.info("***** casual-mask *****")
			if input_ori_ids is not None:
				seq_features['input_ids'] = tf.identity(input_ori_ids)
				tf.logging.info("***** casual-mask ori-input-ids *****")
			else:
				tf.logging.info("***** casual-mask ori-input-ids *****")
				
		if input_ori_ids is not None:
			not_equal = tf.cast(tf.not_equal(input_ori_ids, tf.zeros_like(features["input_ori_ids"])), tf.int32)
			not_equal = tf.reduce_sum(not_equal, axis=-1)
			loss_mask = tf.cast(tf.not_equal(not_equal, tf.zeros_like(not_equal)), tf.float32)
			
			if not kargs.get('use_tpu', False):
				tf.summary.scalar('loss_mask', tf.reduce_sum(loss_mask))

		if not casual_flag and model_config.get("corrupted", True):
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
			tf.logging.info("***** apply random sampling *****")
			seq_features['input_ids'] = output_ids

		model = model_api(model_config, seq_features, labels,
							mode, "", reuse=tf.AUTO_REUSE,
							**kargs)
		model_io_fn = model_io.ModelIO(model_io_config)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope
		
		# if mode == tf.estimator.ModeKeys.TRAIN:
		if kargs.get('mask_type', 'left2right') == 'left2right':
			tf.logging.info("***** using left2right mask and loss *****")
			sequence_mask = tf.to_float(tf.not_equal(features['input_ori_ids'][:, 1:], 
														kargs.get('[PAD]', 0)))
		elif kargs.get('mask_type', 'left2right') == 'seq2seq':
			tf.logging.info("***** using seq2seq mask and loss *****")
			sequence_mask = tf.to_float(features['segment_ids'][:, 1:])
			if not kargs.get('use_tpu', False):
				tf.summary.scalar("loss mask", tf.reduce_mean(sequence_mask))

		# batch x seq_length
		if casual_flag:
			print(model.get_sequence_output_logits().get_shape(), "===logits shape===")
			seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
							labels=features['input_ori_ids'][:, 1:], 
							logits=model.get_sequence_output_logits()[:, :-1])

			per_example_loss = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
			loss = tf.reduce_mean(per_example_loss)

			if model_config.get("cnn_type", "dgcnn") in ['bi_dgcnn', 'bi_light_dgcnn']:
				seq_backward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
							labels=features['input_ori_ids'][:, :-1], 
							logits=model.get_sequence_backward_output_logits()[:, 1:])
			
				per_backward_example_loss = tf.reduce_sum(seq_backward_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
				backward_loss = tf.reduce_mean(per_backward_example_loss)
				loss += backward_loss
				tf.logging.info("***** using backward loss *****")
		elif not casual_flag and model_config.get("corrupted", True):
			(masked_lm_loss,
			masked_lm_example_loss, 
			masked_lm_log_probs,
			masked_lm_mask) = pretrain.seq_mask_masked_lm_output(
										model_config, 
										model.get_sequence_output(), 
										model.get_embedding_table(),
										seq_features['input_mask'], 
										seq_features['input_ori_ids'], 
										seq_features['input_ids'],
										sampled_binary_mask,
										reuse=tf.AUTO_REUSE,
										embedding_projection=model.get_embedding_projection_table())
			loss = masked_lm_loss
			tf.logging.info("***** using masked lm loss *****")
		else:
			masked_lm_positions = features["masked_lm_positions"]
			masked_lm_ids = features["masked_lm_ids"]
			masked_lm_weights = features["masked_lm_weights"]

			(masked_lm_loss,
			masked_lm_example_loss, 
			masked_lm_log_probs,
			masked_lm_mask) = pretrain.get_masked_lm_output(
											model_config, 
											model.get_sequence_output(), 
											model.get_embedding_table(),
											masked_lm_positions, 
											masked_lm_ids, 
											masked_lm_weights,
											reuse=tf.AUTO_REUSE,
											embedding_projection=model.get_embedding_projection_table(),
											pretrain_loss_type="normal")
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
			seq_features['input_ids'] = output_ids

			model = model_api(model_config, seq_features, labels,
								mode, "", reuse=tf.AUTO_REUSE,
								**kargs)

			with tf.variable_scope('cls/discriminator_predictions', reuse=tf.AUTO_REUSE):
				(disc_loss, 
				logits, 
				per_example_loss) = disc_classifier(model_config, 
										model.get_sequence_output(),
										seq_features['input_ori_ids'],
										seq_features['input_ids'],
										seq_features['input_mask'],
										2,
										dropout_prob,
										use_tpu=kargs.get('use_tpu', False),
										sampled_binary_mask=sampled_binary_mask)
			loss += 50.0*disc_loss
			disc_pretrain_tvars = model_io_fn.get_params("cls/discriminator_predictions", 
										not_storage_params=not_storage_params)
			print(disc_pretrain_tvars, '===disc params==')
			tf.logging.info("***** using discriminator_predictions loss *****")
		else:
			disc_pretrain_tvars = []
		

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

		pretrained_tvars.extend(lm_pretrain_tvars)
		pretrained_tvars.extend(disc_pretrain_tvars)

		use_tpu = 1 if kargs.get('use_tpu', False) else 0

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu)
		else:
			scaffold_fn = None

		if mode == tf.estimator.ModeKeys.TRAIN:

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
								use_tpu=use_tpu)

				# train_metric_dict = train_metric(features['input_ori_ids'], 
				# 								model.get_sequence_output_logits(),
				# 								seq_features,
				# 								**kargs)

				# if not kargs.get('use_tpu', False):
				# 	for key in train_metric_dict:
				# 		tf.summary.scalar(key, train_metric_dict[key])
				# 	tf.summary.scalar('learning_rate', optimizer_fn.learning_rate)
				# 	tf.logging.info("***** logging metric *****")
				# 	tf.summary.scalar("causal_attenion_mask_length", tf.reduce_sum(sequence_mask))
					# tf.summary.scalar("bi_attenion_mask_length", tf.reduce_sum(model.bi_attention_mask))

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

			gpu_eval_metrics = eval_metric(features['input_ori_ids'],
										model.get_sequence_output_logits(),
										seq_features,
										**kargs)
			tpu_eval_metrics = (eval_metric, [
										features['input_ori_ids'],
										model.get_sequence_output_logits(),
										seq_features,
										kargs.get('mask_type', 'left2right')
									])	

			if kargs.get('use_tpu', False):
				estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
							  mode=mode,
							  loss=loss,
							  eval_metrics=tpu_eval_metrics,
							  scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=gpu_eval_metrics)

			return estimator_spec

		elif mode == tf.estimator.ModeKeys.PREDICT:
			if kargs.get('predict_type', 'sample_sequence') == 'sample_sequence':
				results = bert_seq_sample_utils.sample_sequence(model_api,
										model_config, 
										mode, 
										features,
										target="", 
										start_token=kargs.get("start_token_id", 101), 
										batch_size=None, 
										context=features.get("context", None), 
										temperature=kargs.get("sample_temp", 1.0), 
										n_samples=kargs.get("n_samples", 1),
										top_k=0,
										end_token=kargs.get("end_token_id", 102),
										greedy_or_sample="greedy",
										gumbel_temp=0.01,
										estimator="stop_gradient",
										back_prop=True,
										swap_memory=True,
										seq_type=kargs.get("seq_type", "seq2seq"),
										mask_type=kargs.get("mask_type", "seq2seq"),
                    					attention_type=kargs.get('attention_type', 'normal_attention')
                    					)
				# stop_gradient output:
				# samples, mask_sequence, presents, logits, final
				
				sampled_token = results['samples']
				sampled_token_logits = results['logits']
				mask_sequence = results['mask_sequence']

				estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'token':sampled_token,
												"logits":sampled_token_logits,
												"mask_sequence":mask_sequence
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'token':sampled_token,
														"logits":sampled_token_logits,
														"mask_sequence":mask_sequence
													}
												)
									}
						)

				return estimator_spec

			elif kargs.get('predict_type', 'sample_sequence') == 'infer_inputs':

				sequence_mask = tf.to_float(tf.not_equal(features['input_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))

				if kargs.get('mask_type', 'left2right') == 'left2right':
					tf.logging.info("***** using left2right mask and loss *****")
					sequence_mask = tf.to_float(tf.not_equal(features['input_ori_ids'][:, 1:], 
																kargs.get('[PAD]', 0)))
				elif kargs.get('mask_type', 'left2right') == 'seq2seq':
					tf.logging.info("***** using seq2seq mask and loss *****")
					sequence_mask = tf.to_float(features['segment_ids'][:, 1:])
					if not kargs.get('use_tpu', False):
						tf.summary.scalar("loss mask", tf.reduce_mean(sequence_mask))

				output_logits = model.get_sequence_output_logits()[:, :-1]
				# output_logits = tf.nn.log_softmax(output_logits, axis=-1)

				output_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=features['input_ids'][:, 1:], 
										logits=output_logits)

				per_example_perplexity = tf.reduce_sum(output_id_logits * sequence_mask, 
												axis=-1) # batch
				per_example_perplexity /= tf.reduce_sum(sequence_mask, axis=-1) # batch

				perplexity = tf.exp(per_example_perplexity)

				estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'token':features['input_ids'][:, 1:],
												"logits":output_id_logits,
												'perplexity':perplexity,
												# "all_logits":output_logits
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'token':features['input_ids'][:,1:],
														"logits":output_id_logits,
														'perplexity':perplexity,
														# "all_logits":output_logits
													}
												)
									}
						)

				return estimator_spec
		else:
			raise NotImplementedError()

	return model_fn
