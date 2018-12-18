from model.bert import bert
from model_io import model_io
# from optimizer import optimizer
from optimizer import hvd_distributed_optimizer as optimizer
from task_module import pretrain, classifier
import tensorflow as tf
from utils.bert import bert_utils
from model.regularizer import vib
from utils.attention import attention_utils
from adversial.adversial_utils import get_perturbation

def base_model(model_config, features, labels, 
			mode, target, perturbation=None, reuse=None):
	
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
						reuse=reuse,
						perturbation=perturbation)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_pooler(reuse=reuse)

	return model

def model_builder_fn(model_config,
				num_labels,
				init_checkpoint,
				model_reuse=None,
				load_pretrained=True,
				model_io_fn=None,
				model_io_config={},
				opt_config={},
				input_name="",
				label_tensor=None,
				exclude_scope="",
				not_storage_params=["adam_m", "adam_v"]):
	
	def model_fn(features, labels, mode):
		label_ids = features["label_ids"]
		
		model = base_model(model_config, features, 
							labels, mode, input_name, 
							reuse=model_reuse,
							perturbation=None)
		
		if mode == tf.estimator.ModeKeys.TRAIN:
			hidden_dropout_prob = model_config.hidden_dropout_prob
			attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
			dropout_prob = model_config.dropout_prob
		else:
			hidden_dropout_prob = 0.0
			attention_probs_dropout_prob = 0.0
			dropout_prob = 0.0

		# assert len(model_lst) == len(input_name)

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		def build_discriminator(model, scope, reuse):

			with tf.variable_scope(scope, reuse=reuse):

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

				(loss, 
				per_example_loss, 
				logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob,
											ratio_weight)
				return loss, per_example_loss, logits

		[loss, per_example_loss, logits] = build_discriminator(
											model,
											scope,
											model_reuse)

		if mode == tf.estimator.ModeKeys.TRAIN:
			pretrained_tvars = model_io_fn.get_params(
									model_config.scope, 
									not_storage_params=not_storage_params)

			if load_pretrained:
				tf.logging.info(" load pre-trained base model ")
				print(" load pre-trained base model ")
				model_io_fn.load_pretrained(
										pretrained_tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)
			tvars = pretrained_tvars
			model_io_fn.set_saver(var_lst=tvars)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer_fn = optimizer.Optimizer(opt_config)
				optimizer_fn.get_opt(
								opt_config.init_lr, 
								opt_config.num_train_steps)

				perturb = get_perturbation(model_config, 
									optimizer_fn.opt, 
									model.embedding_output_word, 
									loss, tvars)

				adv_model = base_model(model_config, features, 
						labels, mode, input_name, 
						reuse=True,
						perturbation=perturb)

				[adv_loss, 
				adv_per_example_loss, 
				adv_logits] = build_discriminator(adv_model,
											scope,
											True)

				total_loss = adv_loss + loss
				total_train_op = optimizer_fn.get_train_op_v1(
								total_loss, tvars)

			return [total_train_op, total_loss, per_example_loss, logits]
		else:
			model_io_fn.set_saver()
			return [loss, loss, per_example_loss, logits]
	return model_fn