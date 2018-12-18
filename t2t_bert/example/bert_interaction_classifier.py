from model.bert import bert
from model_io import model_io
from optimizer import optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils

def base_model(model_config, features, labels, 
			mode, target, reuse=None):
	
	input_ids = features["input_ids_{}".format(target)]
	input_mask = features["input_mask_{}".format(target)]
	segment_ids = features["segment_ids_{}".format(target)]

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = bert.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_pooler(reuse=reuse)

	return model

def classifier_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							model_reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							model_io_config={},
							opt_config={},
							input_name=["a", "b"]):

	def model_fn(features, labels, mode):
		label_ids = features["label_ids"]
		model_lst = []
		for index, name in enumerate(input_name):
			if index > 0:
				reuse = True
			else:
				reuse = model_reuse
			model_lst.append(base_model(model_config, features, 
								labels, mode, name, reuse=reuse))

		if mode == tf.estimator.ModeKeys.TRAIN:
			hidden_dropout_prob = model_config.hidden_dropout_prob
			attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
			dropout_prob = model_config.dropout_prob
		else:
			hidden_dropout_prob = 0.0
			attention_probs_dropout_prob = 0.0
			dropout_prob = 0.0

		assert len(model_lst) == len(input_name)

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		with tf.variable_scope(scope, reuse=model_reuse):
			seq_output_lst = [model.get_pooled_output() for model in model_lst]

			[loss, 
			per_example_loss, 
			logits] = classifier.interaction_classifier(
						model_config, seq_output_lst, 
						num_labels, label_ids,
						dropout_prob)

		# model_io_fn = model_io.ModelIO(model_io_config)
		pretrained_tvars = model_io_fn.get_params(model_config.scope)
		if load_pretrained:
			model_io_fn.load_pretrained(pretrained_tvars, 
										init_checkpoint)

		tvars = model_io_fn.get_params(scope)

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer_fn = optimizer.Optimizer(opt_config)
				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				return [train_op, loss, per_example_loss, logits]
		else:
			model_io_fn.print_params(tvars, string=", trainable params")
			return [loss, loss, per_example_loss, logits]
	return model_fn






			


