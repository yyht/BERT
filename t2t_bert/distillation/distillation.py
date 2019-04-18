import tensorflow as tf
import numpy as np
from system.distillation.distillation_utils import logits_distillation, feature_distillation

class KnowledgeDistillation(object):
	def __init__(self, config={}):
		self.config = config
		self.global_step = tf.train.get_or_create_global_step()

	def _ratio_decay(self, init_ratio, ratio_decay, decay_rate, num_train_steps):
		if ratio_decay == "polynomial_decay":
			ratio_rate = tf.train.polynomial_decay(
													init_ratio,
													self.global_step,
													num_train_steps,
													end_learning_rate=0.0,
													power=1.0,
													cycle=False)
		elif ratio_decay == "cosine_decay":
			ratio_rate = tf.train.cosin_decay(
												init_ratio,
												self.global_step,
												num_train_steps,
												alpha=0.0,
												cycle=False)
		elif ratio_decay == "exponential_decay":
			ratio_rate = tf.train.exponential_decay(
													init_ratio,
													self.global_step,
													num_train_steps,
													decay_rate=decay_rate,
													staircase=False)
		elif ratio_decay == "natural_exp_decay":
			ratio_rate = tf.train.natural_exp_decay(
													init_ratio,
													self.global_step,
													num_train_steps,
													decay_rate=decay_rate,
													staircase=False)
		else:
			ratio_rate = init_ratio
		return ratio_rate

	def distillation(self, features,
					num_labels, dropout_prob, model_reuse,
					num_train_steps, **kargs):

		student_tensor = features["student_tensor"]
		teacher_tensor = features["teacher_tensor"]

		total_distillation_loss = 0.0

		for distillation_type in self.config.get("distillation", ["logits", "feature"]):

			if distillation_type == "logits":
				distillation_loss = logits_distillation(student_tensor, 
											teacher_tensor, 
											self.config.get("kd_type", "kd"))
				distillation_loss *= features["distillation_ratio"]
				distillation_loss = tf.reduce_sum(distillation_loss) / (1e-10+tf.reduce_sum(features["distillation_ratio"]))
				distillation_loss *= self._ratio_decay(kargs.get("logits_ratio", 0.5),
														kargs.get("logits_ratio_decay", "constant"),
														 kargs.get("logits_decay_rate", 0.999)
														num_train_steps)
				total_distillation_loss += distillation_loss


			if distillation_type == "feature":
				student_label = features["student_label"]
				teacher_label = features["teacher_label"]
				with tf.variable_scope(self.config.scope+"/dann_distillation", reuse=model_reuse):
					[student_loss, 
					student_example_loss, 
					student_logits] = feature_distillation(student_tensor, l, 
													student_label, num_labels,
													dropout_prob)

					tf.get_variable_scope().reuse_variables()

					[teacher_loss, 
					teacher_example_loss, 
					teacher_logits] = feature_distillation(teacher_tensor, l, 
													teacher_label, num_labels,
													dropout_prob)

					distillation_loss = (student_loss + teacher_loss) * self._ratio_decay(
														kargs.get("feature_ratio", 0.5),
														kargs.get("feature_ratio_decay", "constant"),
														 kargs.get("feature_decay_rate", 0.999)
														num_train_steps)
					total_distillation_loss += distillation_loss

		return {"distillation_loss":total_distillation_loss,
				"st_logits":student_logits,
				"te_logits":teacher_logits}







