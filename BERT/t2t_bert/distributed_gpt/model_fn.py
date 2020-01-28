import tensorflow as tf
import numpy as np
from distributed_encoder.gpt_encoder import gpt_encoder
from model_io import model_io
from optimizer import distributed_optimizer as optimizer
from model.gpt import sample
from model.gpt import gpt_utils

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

		model = gpt_encoder(model_config, features, labels, 
			mode, target, reuse=tf.AUTO_REUSE)
		scope = model_config.scope

		if mode == tf.estimator.ModeKeys.TRAIN:
			# batch x seq_length
			sequence_mask = tf.to_float(tf.not_equal(features['input_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))

			# batch x seq_length
			seq_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=features['input_ids'][:, 1:], 
						logits=model.get_sequence_output_logits()[:, :-1])

			per_example_loss = tf.reduce_sum(seq_loss*sequence_mask, axis=-1) / (tf.reduce_sum(sequence_mask, axis=-1)+1e-10)
			loss = tf.reduce_mean(per_example_loss)

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

				train_metric_dict = train_metric(features['input_ids'], 
												model.get_sequence_output_logits(), 
												**kargs)

				for key in train_metric_dict:
					tf.summary.scalar(key, train_metric_dict[key])
				tf.summary.scalar('learning_rate', optimizer_fn.learning_rate)
				tf.summary.scalar('seq_length', tf.reduce_mean(tf.reduce_sum(sequence_mask, axis=-1)))

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
			if kargs.get('predict_type', 'sample_sequence') == 'sample_sequence':
				results = sample.sample_sequence(
							gpt_encoder, hparams=model_config, 
							length=kargs.get('max_length', 64), 
							start_token=None, 
							batch_size=10, 
							context=features['input_ids'],
							temperature=2,
							top_k=10)
				
				sampled_token = results['tokens'][:, 1:]
				sampled_token_logits = results['logits'][:, 1:]

				estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'token':sampled_token,
												"logits":sampled_token_logits
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'token':sampled_token,
														"logits":sampled_token_logits
													}
												)
									}
						)

				return estimator_spec

			elif kargs.get('predict_type', 'sample_sequence') == 'infer_inputs':
				sequence_mask = tf.to_float(tf.not_equal(features['input_ids'][:, 1:], 
													kargs.get('[PAD]', 0)))
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
												'perplexity':perplexity
									},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'token':features['input_ids'][:,1:],
														"logits":output_id_logits,
														'perplexity':perplexity
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

