from pai_encoder.bert_encoder import bert_encoder
from model_io import model_io
from task_module import classifier
import tensorflow as tf
from metric import tf_metrics

from utils.bert import bert_utils
from utils.rnn import rnn_utils
from utils.attention import attention_utils

EPSILON = 1e-20

def lstm_model(config, repres, input_mask, 
			dropout_rate, scope, reuse):

	with tf.variable_scope(scope+"/lstm", reuse=reuse):
		tf.logging.info(" lstm scope {}".format(scope+"/lstm"))
		# print(" lstm scope {}".format(scope+"/lstm"))

		shape_lst = bert_utils.get_shape_list(repres, expected_rank=3)

		batch_size = shape_lst[0]
		input_size = shape_lst[-1]

		rnn_kernel = rnn_utils.BiCudnnRNN(config.lstm_dim, batch_size, input_size,
			  num_layers=1, dropout=dropout_rate, kernel='lstm')

		input_lengths = tf.reduce_sum(input_mask, axis=1)

		res, _ , _ = rnn_kernel(repres, 
			 seq_len=tf.cast(input_lengths, tf.int32), 
			 batch_first=True,
			 scope="bidirection_cudnn_rnn",
			reuse=reuse)

		f_rep = res[:, :, 0:config.lstm_dim]
		b_rep = res[:, :, config.lstm_dim:2*config.lstm_dim]
		# print("==lstm output shape==", res.get_shape())
		return res

def alignment_aggerate(config, repres_a, repres_b, 
				repres_mask_a, repres_mask_b, 
				scope, reuse):

	[a_attn, b_attn] = attention_utils.query_context_alignment(repres_a, repres_b, 
				repres_mask_a, repres_mask_b,
				scope+"/alignment", reuse=reuse)
	tf.logging.info(" alignment scope {}".format(scope+"/alignment"))
	# print(" alignment scope {}".format(scope+"/alignment"))
	a_output = tf.concat([repres_a, a_attn, 
						repres_a-a_attn, repres_a*a_attn], axis=-1)
	b_output = tf.concat([repres_b, b_attn,
						repres_b-b_attn, repres_b*b_attn], axis=-1)
	return a_output, b_output

def _split_heads(x, num_heads):
	"""Split channels (dimension 2) into multiple heads,
		becomes dimension 1).
	Must ensure `x.shape[-1]` can be deviced by num_heads
	"""

	shape_lst = bert_utils.get_shape_list(x, expected_rank=3)

	depth = shape_lst[-1]
	# print(x.get_shape(), "===splitheads===")
	splitted_x = tf.reshape(x, [shape_lst[0], shape_lst[1], \
		num_heads, depth // num_heads])
	return tf.transpose(splitted_x, [0, 2, 1, 3])

# def multihead_self_attn(config, repres, repres_mask,
# 						dropout_rate, 
# 						scope, reuse):
# 	ignore_padding = tf.cast(1 - repres_mask, tf.float32)
# 	ignore_padding = attention_utils.attention_bias_ignore_padding(ignore_padding)
# 	encoder_self_attention_bias = ignore_padding
# 	with tf.variable_scope(scope+"/multihead_attention", reuse=reuse):
# 		output = attention_utils.multihead_attention_texar(repres, 
# 						memory=None, 
# 						memory_attention_bias=encoder_self_attention_bias,
# 						num_heads=config.num_heads, 
# 						num_units=None, 
# 						dropout_rate=dropout_rate, 
# 						scope=scope+"/multihead_attention")
# 		tf.logging.info(" alignment aggerate scope {}".format(scope+"/multihead_attention"))
# 		# print(" alignment aggerate scope {}".format(scope+"/multihead_attention"))
# 		# batch x num_head x seq x dim
# 		output = _split_heads(output, config.num_heads)
# 		return output

def bert_encoding(model_config, features, labels, 
			mode, target, max_len,
			scope, dropout_rate, 
			reuse=None):

	model = bert_encoder(model_config, features, labels, 
			mode, target, reuse=reuse)

	repres = model.get_sequence_output()
	input_mask = features["input_mask_{}".format(target)]

	return [input_mask, repres]

def multihead_pooling(config, repres, repres_mask,
						num_units,
						dropout_rate, scope, reuse):
	repres = lstm_model(config, repres, repres_mask, 
			dropout_rate, scope+"/aggerate", reuse)

	shape_lst = bert_utils.get_shape_list(repres, expected_rank=3)

	output = attention_utils.multihead_pooling(repres, 
					sequence_mask=repres_mask,
					num_units=num_units,
					mask_zero=True,
					num_heads=config.num_heads,
					scope_name=scope, 
					reuse=reuse) # batch x dim

	print("---output shape---{}".format(output.get_shape()))

	return output

def ave_max_pooling(config, repres, repres_mask,
				num_units,
				dropout_rate, scope, reuse):

	repres = lstm_model(config, repres, repres_mask, 
			dropout_rate, scope+"/pooling", reuse)

	shape_lst = bert_utils.get_shape_list(repres, expected_rank=3)
	repres_len = tf.reduce_sum(repres_mask, axis=-1)
	repres_sum = tf.reduce_sum(repres, 1)
    repres_ave = tf.div(repres_sum, tf.expand_dims(tf.cast(repres_len, tf.float32)+EPSILON, -1))

    mask = tf.expand_dims(repres_mask, -1)
    repres_max = tf.reduce_max(mask_logits(repres, mask), axis=1)

    out = tf.concat([repres_ave, repres_max], axis=-1)
    return out

def model_fn_builder(
					model_config,
					num_labels,
					init_checkpoint,
					model_reuse=None,
					load_pretrained=True,
					model_io_fn=None,
					optimizer_fn=None,
					model_io_config={},
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target=["a", "b"]):

	def model_fn(features, labels, mode):

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		label_ids = features["label_ids"]

		model_lst = []
		for index, name in enumerate(target):
			if index > 0:
				reuse = True
			else:
				reuse = model_reuse
			model_lst.append(bert_encoding(model_config, features, labels, 
												mode, name,
												scope, dropout_rate, 
												reuse=reuse))

		[input_mask_a, repres_a] = model_lst[0]
		[input_mask_b, repres_b] = model_lst[1]

		output_a, output_b = alignment_aggerate(model_config, 
				repres_a, repres_b, 
				input_mask_a, 
				input_mask_b, 
				scope, 
				reuse=model_reuse)

		if model_config.pooling == "ave_max_pooling":
			pooling_fn = ave_max_pooling
		elif model_config.pooling == "multihead_pooling":
			pooling_fn = multihead_pooling

		repres_a = pooling_fn(model_config, output_a, 
					input_mask_a, 
					scope, 
					dropout_prob, 
					reuse=model_reuse)

		repres_b = pooling_fn(model_config, output_b,
					input_mask_b,
					scope, 
					dropout_prob,
					reuse=True)

		pair_repres = tf.concat([repres_a, repres_b,
					tf.abs(repres_a-repres_b),
					repres_b*repres_a], axis=-1)

		with tf.variable_scope(scope, reuse=model_reuse):
			(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											pair_repres,
											num_labels,
											label_ids,
											dropout_prob)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)
		if load_pretrained:
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		model_io_fn.set_saver(var_lst=tvars)

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer_fn.get_train_op(loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss, train_op=train_op)
				return {
							"estimator_spec":output_spec, 
							"train":{
										"loss":loss, 
										"logits":logits,
										"train_op":train_op
									}
						}
		elif mode == tf.estimator.ModeKeys.PREDICT:
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			max_prob = tf.reduce_max(prob, axis=-1)
			
			estimator_spec = tf.estimator.EstimatorSpec(
									mode=mode,
									predictions={
												'pred_label':pred_label,
												"max_prob":max_prob
								  	},
									export_outputs={
										"output":tf.estimator.export.PredictOutput(
													{
														'pred_label':pred_label,
														"max_prob":max_prob
													}
												)
								  	}
						)
			return {
						"estimator_spec":estimator_spec 
					}
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
									"loss": sentence_mean_loss,
									"acc":sentence_accuracy
								}

				return eval_metric_ops

			eval_metric_ops = metric_fn( 
							per_example_loss,
							logits, 
							label_ids)
			
			estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=eval_metric_ops)
			return {
						"estimator_spec":estimator_spec, 
						"eval":{
							"per_example_loss":per_example_loss,
							"logits":logits
						}
					}
		else:
			raise NotImplementedError()
	return model_fn


