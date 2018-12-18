from model.bert import bert
from model_io import model_io
# from optimizer import optimizer
from optimizer import hvd_distributed_optimizer as optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils
from model.regularizer import vib
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

def classifier_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							model_reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							model_io_config={},
							opt_config={},
							input_name=["a", "b"],
							label_tensor=None):

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
			try:
				label_ratio_table = tf.get_variable(
							name="label_ratio",
							initializer=tf.constant(label_tensor),
							trainable=False)

				ratio_weight = tf.nn.embedding_lookup(label_ratio_table,
				 	label_ids)
				print("==applying class weight==")
			except:
				ratio_weight = None

			seq_output_lst = [model.get_pooled_output() for model in model_lst]

			[loss, 
			per_example_loss, 
			logits] = classifier.order_classifier(
						model_config, seq_output_lst, 
						num_labels, label_ids,
						dropout_prob,ratio_weight)

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

def classifier_model_fn_builder_v1(
							model_config,
							num_labels,
							init_checkpoint,
							model_reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							model_io_config={},
							opt_config={},
							input_name=["a", "b"],
							label_tensor=None):

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

			try:
				print("==applying class weight==")
				label_ratio_table = tf.get_variable(
							name="label_ratio",
							shape=[num_labels,],
							initializer=tf.constant(label_tensor),
							trainable=False)

				ratio_weight = tf.nn.embedding_lookup(label_ratio_table,
				 	label_ids)
			except:
				print("==not applying class weight==")
				ratio_weight = None

			seq_output_lst = [model.get_pooled_output() for model in model_lst]

			[loss, 
			per_example_loss, 
			logits] = classifier.order_classifier_v1(
						model_config, seq_output_lst, 
						num_labels, label_ids,
						dropout_prob,
						ratio_weight)

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

def classifier_vib_model_fn_builder(
							model_config,
							num_labels,
							init_checkpoint,
							model_reuse=None,
							load_pretrained=True,
							model_io_fn=None,
							model_io_config={},
							opt_config={},
							input_name=["a", "b"],
							vib_config={},
							label_tensor=None):

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

			try:
				label_ratio_table = tf.get_variable(
							name="label_ratio",
							shape=[num_labels,],
							initializer=tf.constant(label_tensor),
							trainable=False)

				ratio_weight = tf.nn.embedding_lookup(label_ratio_table,
				 	label_ids)
			except:
				ratio_weight = None

			seq_output_lst = [model.get_pooled_output() for model in model_lst]
			repres = seq_output_lst[0] + seq_output_lst[1]

			final_hidden_shape = bert_utils.get_shape_list(
									repres, 
									expected_rank=2)

			z_mean = tf.layers.dense(repres, final_hidden_shape[1], name="z_mean")
			z_log_var = tf.layers.dense(repres, final_hidden_shape[1], name="z_log_var")
			print("=======applying vib============")
			if mode == tf.estimator.ModeKeys.TRAIN:
				print("====applying vib====")
				vib_connector = vib.VIB(vib_config)
				[kl_loss, 
				latent_vector] = vib_connector.build_regularizer(
											[z_mean, z_log_var])

				[loss, 
				per_example_loss, 
				logits] = classifier.classifier(model_config,
												latent_vector,
												num_labels,
												label_ids,
												dropout_prob,
												ratio_weight)

				loss += tf.reduce_mean(kl_loss)
			else:
				print("====applying z_mean for prediction====")
				[loss, 
				per_example_loss, 
				logits] = classifier.classifier(model_config,
												z_mean,
												num_labels,
												label_ids,
												dropout_prob,
												ratio_weight)

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

def bert_layer_aggerate(encoding_lst, scope, reuse):
	with tf.variable_scope(scope+"/layer_aggerate", reuse=reuse):
		valid_tensor = tf.stack(encoding_lst, axis=1) # batch x num_layer x seq x dim
		attn = tf.get_variable("layer_attention",
										dtype=tf.float32,
										shape=[len(encoding_lst),],
										initializer=tf.initializers.random_uniform(0,1))

		prob = tf.exp(tf.nn.log_softmax(attn))

		layer_epres = tf.einsum("abcd,b->acd", valid_tensor, prob)
		return layer_epres

def bert_seq_aggerate(repres, input_mask, scope, reuse):
	with tf.variable_scope(scope+"/seq_aggerate", reuse=reuse):
		attn = tf.get_variable("seq_attention",
							dtype=tf.float32,
							shape=[tf.shape(repres)[1],],
							initializer=tf.initializers.random_uniform(0,1))

		out = tf.dot(repres, attn) # batch x seq
		masked_out = attention_utils.mask_logits(out, input_mask)

		weight = tf.exp(tf.nn.log_softmax(masked_out, axis=1)) # batch x seq
		weight = tf.expand_dims(weight, -1) # batch x seq x 1
		seq_repres = tf.reduce_sum(weight * out, axis=1)
		return seq_repres

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
							label_tensor=None):

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
			try:
				print("==applying class weight==")
				label_ratio_table = tf.get_variable(
							name="label_ratio",
							shape=[num_labels,],
							initializer=tf.constant(label_tensor),
							trainable=False)

				ratio_weight = tf.nn.embedding_lookup(label_ratio_table,
				 	label_ids)
			except:
				ratio_weight = None

			seq_output_lst = []
			for name, model in zip(input_name, model_lst):
				layers = model.get_all_encoder_layers
				layer_repres = bert_layer_aggerate(layers, scope, model_reuse)
				input_mask = features["input_mask_{}".format(name)]
				seq_repres = bert_seq_aggerate(layer_repres, input_mask, scope, model_reuse)
				seq_output_lst.append(seq_repres)

			[loss, 
			per_example_loss, 
			logits] = classifier.order_classifier(
						model_config, seq_output_lst, 
						num_labels, label_ids,
						dropout_prob,ratio_weight)

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






			


