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

from task_module import classifier
from task_module import tsa_pretrain
import tensorflow as tf
from metric import tf_metrics

def train_metric(input_ids, predicted_logits, **kargs):
	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = predicted_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	sequence_mask = tf.to_float(tf.not_equal(labels, 
								kargs.get('[PAD]', 0)))

	per_example_perplexity = tf.reduce_sum(input_id_logits * sequence_mask, axis=-1) # batch
	per_example_perplexity /= tf.reduce_sum(sequence_mask, axis=-1) # batch

	perplexity = tf.reduce_mean(tf.exp(per_example_perplexity))

	lm_token_accuracy = tf.equal(
						tf.cast(labels, tf.int32),
						tf.cast(tf.argmax(logits, axis=-1), tf.int32))

	lm_token_accuracy = tf.reduce_sum(tf.cast(lm_token_accuracy, tf.float32) * sequence_mask, axis=-1)
	lm_token_accuracy /= tf.reduce_sum(sequence_mask, axis=-1) # batch

	return {
		"perplexity": perplexity,
		"token_acc": tf.reduce_mean(lm_token_accuracy)
		}

def eval_metric(input_ids, predicted_logits, **kargs):
	labels = input_ids[:, 1:] # <S>,1,2,3,<T>,<PAD>, <PAD>
	logits = predicted_logits[:, :-1] # 1,2,3,<T>, xxx, xxx

	input_id_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(
										labels=labels, 
										logits=logits)

	sequence_mask = tf.to_float(tf.not_equal(labels, 
								kargs.get('[PAD]', 0)))

	per_example_perplexity = tf.reduce_sum(input_id_logits * sequence_mask, axis=-1) # batch
	per_example_perplexity /= tf.reduce_sum(sequence_mask, axis=-1) # batch

	perplexity = tf.reduce_mean(tf.exp(per_example_perplexity))

	ppl_avg = tf.metrics.mean(values=per_example_perplexity)
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

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)
		
		seq_features = {}
		for key in features:
			seq_features[key] = features[key]
		seq_features['input_ids'] = features["input_ori_ids"]

		model = model_api(model_config, seq_features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope
		
		sequence_mask = tf.to_float(tf.not_equal(features['input_ori_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))

		# batch x seq_length
		print(model.get_sequence_output_logits().get_shape(), "===logits shape===")
		seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=features['input_ori_ids'][:, 1:], 
					logits=model.get_sequence_output_logits()[:, :-1])

		per_example_loss = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
		loss = tf.reduce_mean(per_example_loss)
		
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

		pretrained_tvars.extend(lm_pretrain_tvars)

		use_tpu = 1 if kargs.get('use_tpu', False) else 0

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			scaffold_fn = model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu)
			tf.logging.info("***** using tpu *****")
		else:
			scaffold_fn = None
			tf.logging.info("***** not using tpu *****")

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

				train_metric_dict = train_metric(features['input_ori_ids'], 
												model.get_sequence_output_logits(), 
												**kargs)

				if not kargs.get('use_tpu', False):
					for key in train_metric_dict:
						tf.summary.scalar(key, train_metric_dict[key])
					tf.summary.scalar('learning_rate', optimizer_fn.learning_rate)
					tf.logging.info("***** logging metric *****")
					tf.summary.scalar("causal_attenion_mask_length", tf.reduce_sum(model.attention_mask))
					tf.summary.scalar("bi_attenion_mask_length", tf.reduce_sum(model.bi_attention_mask))

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
										model.get_sequence_output_logits())
			tpu_eval_metrics = (eval_metric, [
										features['input_ori_ids'],
										model.get_sequence_output_logits()
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
		else:
			raise NotImplementedError()

	return model_fn
