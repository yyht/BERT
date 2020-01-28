from model.bert import bert
from model_io import model_io
from optimizer import optimizer
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
							model_io_config={},
							opt_config={}):

	def model_fn(features, labels, mode):
	
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
		pretrained_tvars = model_io_fn.get_params(model_config.scope)
		if load_pretrained:
			model_io_fn.load_pretrained(pretrained_tvars, 
										init_checkpoint)

		tvars = model_io_fn.get_params(scope)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn.print_params(tvars, string=", trainable params")
			with tf.control_dependencies(update_ops):
				optimizer_fn = optimizer.Optimizer(opt_config)
				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

		print(logits.get_shape(), "===logits shape===")
		pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
		prob = tf.nn.softmax(logits)
		max_prob = tf.reduce_max(prob, axis=-1)

		output_spec = tf.estimator.EstimatorSpec(
		  mode=mode,
		  predictions={
			'pred_label':pred_label,
			"max_prob":max_prob
		  },
		  export_outputs={
			"output":tf.estimator.export.PredictOutput(
					{
						'pred_label':pred_label,
						# "true_label_ids":label_ids,
						"max_prob":max_prob
		  			}
		  	)
		  }
		)
		return output_spec
	return model_fn