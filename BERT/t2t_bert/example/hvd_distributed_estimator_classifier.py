from model.bert import bert
from model_io import model_io
from optimizer import hvd_distributed_optimizer as optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils

def classifier_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							optimizer_fn=None,
							model_io_config={},
							opt_config={},
							exclude_scope="",
							not_storage_params=[]):

	def model_fn(features, labels, mode):
		print(features)
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			hidden_dropout_prob = model_config.hidden_dropout_prob
			attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
			dropout_prob = model_config.dropout_prob
		else:
			hidden_dropout_prob = 0.0
			attention_probs_dropout_prob = 0.0
			dropout_prob = 0.0

		model = bert.Bert(model_config)
		model.build_embedder(input_ids, segment_ids,
											hidden_dropout_prob,
											attention_probs_dropout_prob,
											reuse=reuse)
		model.build_encoder(input_ids,
											input_mask,
											hidden_dropout_prob, 
											attention_probs_dropout_prob,
											reuse=reuse)
		model.build_pooler(reuse=reuse)

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		with tf.variable_scope(scope, reuse=reuse):
			(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob)

		# model_io_fn = model_io.ModelIO(model_io_config)
		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)
		if load_pretrained:
			model_io_fn.load_pretrained(pretrained_tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		tvars = pretrained_tvars
		model_io_fn.set_saver(var_lst=tvars)

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				# optimizer_fn = optimizer.Optimizer(opt_config)
				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				return tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss, train_op=train_op)

		elif mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {
			        # Generate predictions (for PREDICT and EVAL mode)
			        "classes": tf.argmax(input=logits, axis=1),
			        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			        # `logging_hook`.
			        "probabilities": 
			        	tf.exp(tf.nn.log_softmax(logits, name="softmax_tensor"))
			    }
			return tf.estimator.EstimatorSpec(mode=mode, 
								predictions=predictions)
		elif mode == tf.estimator.ModeKeys.EVAL:
			"""
			needs to manaually write metric ops
			see 
			https://github.com/google/seq2seq/blob/7f485894d412e8d81ce0e07977831865e44309ce/seq2seq/metrics/metric_specs.py
			"""
			return None