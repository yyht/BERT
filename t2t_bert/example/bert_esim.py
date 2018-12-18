from model.bert import bert
from model_io import model_io
# from optimizer import optimizer
from optimizer import hvd_distributed_optimizer as optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils

from utils.rnn import rnn_utils
from utils.attention import attention_utils

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

def bert_layer_aggerate(encoding_lst, max_len, scope, reuse):
	with tf.variable_scope(scope, reuse=reuse):
		valid_tensor = tf.stack(encoding_lst, axis=1) # batch x num_layer x seq x dim
		attn = tf.get_variable(scope+"/layer_attention",
										dtype=tf.float32,
										shape=[len(encoding_lst),],
										initializer=tf.initializers.random_uniform(0,1))

		prob = tf.exp(tf.nn.log_softmax(attn))

		layer_repres = tf.einsum("abcd,b->acd", valid_tensor, prob)
		# layer_repres = encoding_lst[-1]
		# since input_target_a means b->a 
		# and input_target_b means a->b
		
		layer_repres = layer_repres[:,0:max_len,:]
		
		# print(" bert layer output shape w{}".format(layer_repres.get_shape()))
		return layer_repres

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

def alignment(config, repres_a, repres_b, 
				repres_mask_a, repres_mask_b, 
				scope, max_len, reuse):
	repres_mask_a = repres_mask_a[:,0:max_len]
	repres_mask_b = repres_mask_b[:,0:max_len]
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

def alignment_aggerate_v1(config, repres, repres_mask,
						dropout_rate, 
						scope, reuse):
	ignore_padding = tf.cast(1 - repres_mask, tf.float32)
	ignore_padding = attention_utils.attention_bias_ignore_padding(ignore_padding)
	encoder_self_attention_bias = ignore_padding
	with tf.variable_scope(scope+"/multihead_attention", reuse=reuse):
		output = attention_utils.multihead_attention_texar(repres, 
						memory=None, 
						memory_attention_bias=encoder_self_attention_bias,
						num_heads=config.num_heads, 
						num_units=None, 
						dropout_rate=dropout_rate, 
						scope=scope+"/multihead_attention")
		tf.logging.info(" alignment aggerate scope {}".format(scope+"/multihead_attention"))
		# print(" alignment aggerate scope {}".format(scope+"/multihead_attention"))
		# batch x num_head x seq x dim
		output = _split_heads(output, config.num_heads)
		return output

def generalized_pooling_v1(config, repres, repres_mask,
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

def generalized_pooling(config, repres, repres_mask,
						scope, reuse):

	repres_lst = tf.unstack(repres, axis=1) # [(batch x seq x dim)] * num_head
	# print(repres.get_shape(), "======generalized pooling=====")
	shape_lst = bert_utils.get_shape_list(repres, expected_rank=4)

	encode_dim = shape_lst[-1]
	feature_dim = encode_dim
	attention_dim = encode_dim
	seq_length = tf.cast(tf.reduce_sum(repres_mask, axis=-1), tf.int32)
	maxlen = tf.cast(tf.reduce_max(seq_length), tf.int32)

	pooled_lst = []
	for i, tensor in enumerate(repres_lst):
		# print(tensor.get_shape(), "===geenralized pooling tensor shape")
		pool_scope = scope + "/vector_attention_{}".format(i)
		tf.logging.info(" generalized pooling scope {}".format(pool_scope))
		# print(" generalized pooling scope {}".format(pool_scope))
		pooled_weight = attention_utils.vector_attention(
					tensor, encode_dim, 
					feature_dim, attention_dim, 
					sequence_mask=repres_mask,
                    mask_zero=True, maxlen=maxlen, 
                    epsilon=1e-8, seed=0,
                    scope_name=pool_scope, 
                    reuse=reuse)
		pooled_result = tf.reduce_sum(pooled_weight*tensor, axis=1)
		pooled_lst.append(pooled_result) # batch x dim
	pooled_tensor = tf.concat(pooled_lst, axis=-1) # batch x (dim x num_head)
	return pooled_tensor

def esim_bert_encoding(model_config, features, labels, 
			mode, target, max_len, scope, dropout_rate, 
			reuse=None):

	model = base_model(model_config, features, labels, 
			mode, target, reuse=reuse)

	layers = model.get_all_encoder_layers()
	layer_repres = bert_layer_aggerate(layers, 
					max_len,
					scope, reuse)

	input_mask = features["input_mask_{}".format(target)]
	
	input_mask = input_mask[:,0:max_len]

	repres = lstm_model(model_config, layer_repres, input_mask, 
			dropout_rate, scope, reuse)

	return repres

def esim_bert_pooling(model_config, repres, repres_mask, 
					scope, max_len, dropout_rate, reuse=None):
	# align_aggerate = alignment_aggerate(model_config, repres, repres_mask,
	# 					dropout_rate, 
	# 					scope, reuse)
	repres_mask = repres_mask[:,0:max_len]
	pooled_out = generalized_pooling_v1(model_config, 
						repres, repres_mask,
						model_config.num_units,
						dropout_rate, scope, reuse)
	return pooled_out

def classifier_attn_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							model_reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							model_io_config={},
							opt_config={},
							input_name=["a", "b"],
							label_tensor=None,
							exclude_scope_dict={},
							not_storage_params=[],
							max_len=64,
							**kargs):

	def model_fn(features, labels, mode):

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "/task"
		else:
			scope = model_config.scope

		if mode == tf.estimator.ModeKeys.TRAIN:
			hidden_dropout_prob = model_config.hidden_dropout_prob
			attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
			dropout_prob = model_config.dropout_prob
		else:
			hidden_dropout_prob = 0.0
			attention_probs_dropout_prob = 0.0
			dropout_prob = 0.0

		label_ids = features["label_ids"]
		repres_lst = {}
		for index, name in enumerate(input_name):
			if index > 0:
				reuse = True
			else:
				reuse = model_reuse

			repres_lst[name] = esim_bert_encoding(model_config, 
								features, 
								labels, mode, name, 
								max_len+2,
								scope, dropout_prob,
								reuse=reuse)

		a_output, b_output = alignment(model_config, 
				repres_lst["a"], repres_lst["b"], 
				features["input_mask_{}".format("a")], 
				features["input_mask_{}".format("b")], 
				scope, max_len+2, reuse=model_reuse)

		repres_a = esim_bert_pooling(model_config, a_output, 
					features["input_mask_{}".format("a")], 
					scope, 
					max_len+2,
					dropout_prob, 
					reuse=model_reuse)

		repres_b = esim_bert_pooling(model_config, b_output,
					features["input_mask_{}".format("b")],
					scope, 
					max_len+2,
					dropout_prob,
					reuse=True)

		pair_repres = tf.concat([repres_a, repres_b,
					tf.abs(repres_a-repres_b),
					repres_b*repres_a], axis=-1)

		print(pair_repres.get_shape(), "==repres shape==")

		with tf.variable_scope(scope, reuse=model_reuse):

			try:
				label_ratio_table = tf.get_variable(
							name="label_ratio",
							shape=[num_labels,],
							initializer=tf.constant(label_tensor),
							trainable=False)

				ratio_weight = tf.nn.embedding_lookup(label_ratio_table,
				 	label_ids)
				print("==applying class weight==")
			except:
				ratio_weight = None

			(loss, 
			per_example_loss, 
			logits) = classifier.classifier(model_config,
										pair_repres,
										num_labels,
										label_ids,
										dropout_prob,
										ratio_weight)
		if mode == tf.estimator.ModeKeys.TRAIN:
			pretrained_tvars = model_io_fn.get_params(model_config.scope, 
											not_storage_params=not_storage_params)

			if load_pretrained:
				model_io_fn.load_pretrained(pretrained_tvars, 
											init_checkpoint,
											exclude_scope=exclude_scope_dict["task"])

		trainable_params = model_io_fn.get_params(scope, 
											not_storage_params=not_storage_params)

		tvars = trainable_params

		storage_params = model_io_fn.get_params(model_config.scope, 
											not_storage_params=not_storage_params)

		# for var in storage_params:
		# 	print(var.name, var.get_shape(), "==storage params==")

		# for var in tvars:
		# 	print(var.name, var.get_shape(), "==trainable params==")

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

