from knowledge_distillation import teacher, student
import tensorflow as tf
import numpy as np

from model_io import model_io
from optimizer import optimizer

def distillation_model_fn(model_config_dict,
				num_labels,
				init_checkpoint_dict,
				model_reuse=None,
				load_pretrained={},
				model_io_fn=None,
				model_io_config={},
				opt_config={},
				student_input_name=["a", "b"],
				teacher_input_name=["a", "b"],
				unlabel_input_name=["ua", "ub"],
				temperature=2.0,
				exclude_scope_dict={"student":"", "teacher":""},
				not_storage_params=["adam_m", "adam_v"],
				distillation_weight={},
				if_distill_unlabeled=False):

	def model_fn(features, labels, mode):
		labeled_student_model = student.model_builder_fn(
							model_config_dict["student"],
							num_labels,
							init_checkpoint_dict["student"],
							model_reuse=model_reuse,
							load_pretrained=load_pretrained,
							model_io_fn=model_io_fn,
							model_io_config=model_io_config,
							opt_config=opt_config,
							input_name=student_input_name,
							temperature=temperature,
							exclude_scope=exclude_scope_dict["student"],
							not_storage_params=not_storage_params)

		[loss, per_example_loss, 
		logits, temperature_log_prob] = labeled_student_model(features, 
															labels, 
															mode)
		train_loss = loss * distillation_weight.get("true_label", 1.0)
		tf.logging.info(" build student model ")
		if mode == tf.estimator.ModeKeys.TRAIN:
			tf.logging.info(" build teacher model for training ")
			labeled_teacher_model = teacher.model_builder_fn(
								model_config_dict["teacher"],
								num_labels,
								init_checkpoint_dict["teacher"],
								model_reuse=None,
								load_pretrained=load_pretrained,
								model_io_fn=model_io_fn,
								model_io_config=model_io_config,
								opt_config=opt_config,
								input_name=teacher_input_name,
								temperature=temperature,
								exclude_scope=exclude_scope_dict["teacher"],
								not_storage_params=not_storage_params)

			[tloss, tper_example_loss, 
			tlogits, ttemperature_log_prob] = labeled_teacher_model(features, 
																labels, 
																mode)

			cross_entropy = temperature_log_prob * tf.stop_gradient(tf.exp(ttemperature_log_prob))
			print("===size of cross entropy===", cross_entropy.get_shape())
			distillation_loss = -tf.reduce_sum(cross_entropy, axis=-1)

			distillation_loss = tf.reduce_mean(distillation_loss)
			train_loss += distillation_weight["label"] * distillation_loss

			if if_distill_unlabeled:
				tf.logging.info(" build unlabeled student and teacher model for training ")
				unlabeled_student_model = student.model_builder_fn(
								model_config_dict["student"],
								num_labels,
								init_checkpoint_dict["student"],
								model_reuse=True,
								load_pretrained=load_pretrained,
								model_io_fn=model_io_fn,
								model_io_config=model_io_config,
								opt_config=opt_config,
								input_name=unlabel_input_name,
								temperature=temperature,
								exclude_scope=exclude_scope_dict["student"],
								not_storage_params=not_storage_params)

				[uloss, uper_example_loss, 
				ulogits, utemperature_log_prob] = unlabeled_student_model(features, 
																	labels, 
																	mode)

				unlabeled_teacher_model = teacher.model_builder_fn(
									model_config_dict["teacher"],
									num_labels,
									init_checkpoint_dict["teacher"],
									model_reuse=True,
									load_pretrained=load_pretrained,
									model_io_fn=model_io_fn,
									model_io_config=model_io_config,
									opt_config=opt_config,
									input_name=unlabel_input_name,
									temperature=temperature,
									exclude_scope=exclude_scope_dict["teacher"],
									not_storage_params=not_storage_params)

				[utloss, utper_example_loss, 
				utlogits, uttemperature_log_prob] = unlabeled_teacher_model(features, 
																	labels, 
																	mode)

				cross_entropy = utemperature_log_prob * tf.stop_gradient(tf.exp(uttemperature_log_prob))
				unlabeled_distillation_loss = -tf.reduce_sum(cross_entropy, axis=-1)
				unlabeled_distillation_loss = tf.reduce_mean(unlabeled_distillation_loss)

				train_loss += distillation_weight["unlabel"]*unlabeled_distillation_loss

			teacher_pretrained_tvars = model_io_fn.get_params(
											model_config_dict["teacher"].scope, 
											not_storage_params=not_storage_params)
			if load_pretrained.get("teacher", True):
				tf.logging.info(" load pre-trained teacher model ")
				model_io_fn.load_pretrained(
										teacher_pretrained_tvars, 
										init_checkpoint_dict["teacher"],
										exclude_scope=exclude_scope_dict["teacher"])

		student_pretrained_tvars = model_io_fn.get_params(
										model_config_dict["student"].scope, 
										not_storage_params=not_storage_params)
		if load_pretrained.get("student", True):
			tf.logging.info(" load pre-trained student model ")
			model_io_fn.load_pretrained(
									student_pretrained_tvars, 
									init_checkpoint_dict["student"],
									exclude_scope=exclude_scope_dict["student"])

		tvars = student_pretrained_tvars
		model_io_fn.set_saver(var_lst=tvars)

		if mode == tf.estimator.ModeKeys.TRAIN:
			model_io_fn.print_params(tvars, string=", trainable params")
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer_fn = optimizer.Optimizer(opt_config)
				train_op = optimizer_fn.get_train_op(train_loss, tvars, 
								opt_config.init_lr, 
								opt_config.num_train_steps)

				return [train_op, loss, per_example_loss, logits]
		else:
			model_io_fn.print_params(tvars, string=", trainable params")
			return [loss, loss, per_example_loss, logits]
	return model_fn




