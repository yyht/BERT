from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

from optimizer import optimizer_utils
from optimizer import radam_utils

class Optimizer(object):
	def __init__(self, config, **kargs):
		self.config = config
		for key in self.config:
			print(key, self.config[key], "==opt config==")
		self.global_step = tf.train.get_or_create_global_step()

		num_warmup_steps = self.config.num_warmup_steps
		global_steps_int = tf.cast(self.global_step, tf.int32)
		warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

		self.decay_global_step = tf.cond(global_steps_int < warmup_steps_int,
									lambda:tf.cast(tf.constant(0), tf.int64),
									lambda:self.global_step-tf.cast(warmup_steps_int, tf.int64))

	def lr_decay_fn(self, init_lr, num_train_steps,
					**kargs):
		lr_decay = self.config.get("lr_decay", "polynomial_decay")
		tf.logging.info(" lr decay method {}".format(lr_decay))
		learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
		if lr_decay == "polynomial_decay":
			learning_rate = tf.train.polynomial_decay(
													learning_rate,
													self.global_step,
													num_train_steps,
													end_learning_rate=0.0,
													power=1.0,
													cycle=False)
		elif lr_decay == "cosine_decay":
			learning_rate = tf.train.cosin_decay(
													learning_rate,
													self.global_step,
													num_train_steps,
													alpha=0.0,
													cycle=False)
		elif lr_decay == "exponential_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.exponential_decay(
													learning_rate,
													self.global_step,
													num_train_steps,
													decay_rate=decay_rate,
													staircase=False)
		elif lr_decay == "natural_exp_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.natural_exp_decay(
													learning_rate,
													self.global_step,
													num_train_steps,
													decay_rate=decay_rate,
													staircase=False)
		else:
			learning_rate = learning_rate
		return learning_rate

	def private_global_step(self, global_step):
		global_step = tf.cast(global_step, tf.int64)
		cond_fn = tf.less(self.global_step, tf.constant(self.config.num_warmup_steps, dtype=tf.int64))

		decay_global_step = tf.cond(cond_fn,
									lambda:tf.constant(value=0, shape=[], dtype=tf.int64, name="initial_global_step"),
									lambda:global_step-tf.constant(self.config.num_warmup_steps, dtype=tf.int64))

		return decay_global_step

	def private_lr_decay_fn(self, init_lr, num_train_steps,
							global_step,
							**kargs):
		lr_decay = self.config.get("lr_decay", "polynomial_decay")
		tf.logging.info(" lr decay method {}".format(lr_decay))
		learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32, name="init_lr")
		end_learning_rate = self.config.get("end_learning_rate", 0.0)

		decay_global_step = self.private_global_step(global_step)

		if lr_decay == "polynomial_decay":
			learning_rate = tf.train.polynomial_decay(
													init_lr,
													decay_global_step,
													num_train_steps-self.config.num_warmup_steps,
													end_learning_rate=end_learning_rate,
													power=1.0,
													cycle=False)
		elif lr_decay == "cosine_decay":
			learning_rate = tf.train.cosin_decay(
												learning_rate,
												decay_global_step,
												num_train_steps-self.config.num_warmup_steps,
												alpha=0.0,
												cycle=False)
		elif lr_decay == "exponential_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.exponential_decay(
													learning_rate,
													decay_global_step,
													num_train_steps-self.config.num_warmup_steps,
													decay_rate=decay_rate,
													staircase=False)
		elif lr_decay == "natural_exp_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.natural_exp_decay(
													learning_rate,
													decay_global_step,
													num_train_steps-self.config.num_warmup_steps,
													decay_rate=decay_rate,
													staircase=False)
		else:
			learning_rate = learning_rate
		return learning_rate

	def warm_up(self, learning_rate, init_lr, **kargs):
		num_warmup_steps = self.config.num_warmup_steps
		global_steps_int = tf.cast(self.global_step, tf.int32)
		warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

		global_steps_float = tf.cast(global_steps_int, tf.float32)
		warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

		warmup_percent_done = global_steps_float / warmup_steps_float
		warmup_learning_rate = init_lr * warmup_percent_done

		is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
		learning_rate = (
				(1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
		return learning_rate

	def private_warm_up(self, learning_rate, init_lr, global_step, **kargs):
		num_warmup_steps = self.config.num_warmup_steps
		global_steps_int = tf.cast(global_step, tf.int32)
		warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

		global_steps_float = tf.cast(global_steps_int, tf.float32)
		warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

		warmup_percent_done = global_steps_float / warmup_steps_float
		warmup_learning_rate = init_lr * warmup_percent_done

		is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
		learning_rate = (
				(1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
		return learning_rate

	def grad_clip_fn(self, loss, tvars, **kargs):
		grads = tf.gradients(loss, tvars)
		grad_clip = self.config.get("grad_clip", "global_norm")
		tf.logging.info(" gradient clip method {}".format(grad_clip))
		if grad_clip == "global_norm":
			clip_norm = self.config.get("clip_norm", 1.0)
			[grads, _] = tf.clip_by_global_norm(grads, 
								clip_norm=clip_norm)
		elif grad_clip == "norm":
			clip_norm = self.config.get("clip_norm", 1.0)
			grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
		elif grad_clip == "value":
			clip_min_value = self.config.get("clip_min_value", -1.0)
			clip_max_value = self.config.get("clip_max_value", 1.0)
			grads = [tf.clip_by_value(grad, clip_norm) for grad in grads]
		else:
			grads = grads
		return grads

	def optimizer_op(self, learning_rate,
							**kargs):
		opt_type = kargs.get('train_op', None)
		if opt_type is None:
			opt_type = self.config.get("train_op", "adam_decay")
		tf.logging.info(" optimization method {}".format(opt_type))
		if opt_type not in ["adam_decay", "adam", "lamb_v2", 
								"lamb_v1", "radam",
								"adafactor", "sgd"]:
			raise NotImplementedError()
		if opt_type == "adam_decay":
			opt = optimizer_utils.AdamWeightDecayOptimizer(
						learning_rate=learning_rate,
						weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
						beta_1=self.config.get("beta_1", 0.9),
						beta_2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6),
						bias_correction=self.config.get('bias_correction', False),
						exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
						include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
			tf.logging.info("***** apply adam_decay *****")
		elif opt_type == "adam":
			opt = tf.train.AdamOptimizer(learning_rate,
										beta1=self.config.get("beta_1", 0.9),
										beta2=self.config.get("beta_2", 0.999),
										epsilon=self.config.get("epsilon", 1e-6))
			tf.logging.info("***** apply adam *****")
		elif opt_type == "lamb_v2":
			opt = optimizer_utils.LAMBOptimizer_v2(learning_rate,
							   weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
							   beta_1=self.config.get("beta_1", 0.9),
							   beta_2=self.config.get("beta_2", 0.999),
							   epsilon=self.config.get("epsilon", 1e-6),
							   exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
							   exclude_from_layer_adaptation=None,
							   include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
							   name="LAMBOptimizer")
			tf.logging.info("***** apply lamb_v2 *****")
		elif opt_type == "lamb_v1":
			opt = optimizer_utils.LAMBOptimizer_v1(learning_rate,
							   weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
							   beta_1=self.config.get("beta_1", 0.9),
							   beta_2=self.config.get("beta_2", 0.999),
							   epsilon=self.config.get("epsilon", 1e-6),
							   exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
							   include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
							   name="LAMBOptimizer")
			tf.logging.info("***** apply lamb_v1 *****")
		elif opt_type == 'adad_belief':
			opt = optimizer_utils.AdamBeliefWeightDecayOptimizer(learning_rate,
							   weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
							   beta_1=self.config.get("beta_1", 0.9),
							   beta_2=self.config.get("beta_2", 0.999),
							   epsilon=self.config.get("epsilon", 1e-6),
							   exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
							   include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"],
							   name="AdamBeliefWeightDecayOptimizer")
			tf.logging.info("***** apply adam_belief *****")
		elif opt_type == 'radam':
			opt = radam_utils.RAdamOptimizer(learning_rate=learning_rate,
						 beta1=self.config.get("beta_1", 0.9),
						 beta2=self.config.get("beta_2", 0.999),
						 epsilon=self.config.get("epsilon", 1e-6),
						 weight_decay=self.config.get("opt_decay_rate", 0.01),
						 amsgrad=False,
						 total_steps=config['num_train_steps'],
						 warmup_proportion=0.1,
						 min_lr=0.,
						 use_locking=False,
						 exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
						 include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
			tf.logging.info("***** apply radam *****")
		elif opt_type == "adafactor":
			tf.logging.info("***** apply adafactor *****")
			opt =  optimizer_utils.AdaFactorOptimizer(
								learning_rate=learning_rate,
								weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
								beta_1=self.config.get("beta_1", 0.9),
								beta_2=self.config.get("beta_2", 0.999),
								epsilon=self.config.get("epsilon", 1e-6),
								exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
								include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
		elif opt_type == "sgd":
			tf.logging.info("***** apply sgd *****")
			opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		return opt

	def get_train_op(self, loss, tvars, init_lr, 
							num_train_steps, **kargs):
		tf.logging.info("****** optimizer learning rate ******* %s", str(init_lr))
		if kargs.get('train_op', 'adam_decay') != "radam":
			self.learning_rate = self.lr_decay_fn(init_lr, num_train_steps, **kargs)
			self.learning_rate = self.warm_up(self.learning_rate, init_lr, **kargs)
		grads = self.grad_clip_fn(loss, tvars, **kargs)
		opt = self.optimizer_op(self.learning_rate, **kargs)
		if kargs.get("use_tpu", 0) == 1:
			tf.logging.info("***** Using tpu cross shard optimizer *****")
			opt = tf.contrib.tpu.CrossShardOptimizer(opt)
		train_op = opt.apply_gradients(
					zip(grads, tvars), global_step=self.global_step)
		if kargs.get('train_op', 'adam_decay') in ['adam_decay', 'lamb_v2', 'lamb_v1', 'adafactor']:
			new_global_step = self.global_step + 1
			train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
			return train_op
		else:
			return train_op

	def get_layer_wise_train_op(self, loss, tvars, init_lr, num_train_steps, **kargs):

		if kargs.get('train_op', 'adam_decay') != "radam":
			self.learning_rate = self.lr_decay_fn(init_lr, num_train_steps, **kargs)
			self.learning_rate = self.warm_up(self.learning_rate, init_lr, **kargs)

		if kargs.get('train_op', 'adam_decay') in ['adam_decay', 'lamb_v2', 'lamb_v1', 'adafactor']:

			layerwise_lr_decay_power = kargs.get('layerwise_lr_decay_power', 0)
			n_transformer_layers = kargs.get('n_transformer_layers', 4)

			if layerwise_lr_decay_power > 0:
				self.learning_rate = optimizer_utils._get_layer_lrs(self.learning_rate, layerwise_lr_decay_power,
																n_transformer_layers)

			grads = self.grad_clip_fn(loss, tvars, **kargs)
			opt = self.optimizer_op(self.learning_rate, **kargs)
			if kargs.get("use_tpu", 0) == 1:
				tf.logging.info("***** Using tpu cross shard optimizer *****")
				opt = tf.contrib.tpu.CrossShardOptimizer(opt)

			if isinstance(self.learning_rate, dict):
				key_to_grads_and_vars = {}
				for grad, var in zip(grads, tvars):
					update_for_var = False
					for key in self.learning_rate:
						if key in var.name:
							update_for_var = True
							if key not in key_to_grads_and_vars:
								key_to_grads_and_vars[key] = []
							key_to_grads_and_vars[key].append((grad, var))
					if not update_for_var:
						raise ValueError("No learning rate specified for variable", var)
				assignments = []
				for key, key_grads_and_vars in key_to_grads_and_vars.items():
					assignments += opt.apply_gradients(
						key_grads_and_vars, global_step=self.global_step, 
										learning_rate=self.learning_rate[key])
				train_op = tf.group(*assignments, name="layer_wise_lr")
				
				new_global_step = self.global_step + 1
				train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
				return train_op
			else:
				return self.get_train_op(loss, tvars, init_lr, 
							num_train_steps, **kargs)
		else:
			return self.get_train_op(loss, tvars, init_lr, 
							num_train_steps, **kargs)

	def collect_common_vars(self, loss_dict, tvars_dict):
		pass

	def get_group_train_op(self, loss_dict, tvars_dict, init_lr_dict,
							optimizer_type_dict,
							num_train_steps, **kargs):
		opt_list = []
		optimizer_dict = {}

		for key in loss_dict:
			init_lr = init_lr_dict[key]
			optimizer_type = optimizer_type_dict[key]
			if optimizer_type != 'radam':
				learning_rate = self.lr_decay_fn(init_lr, num_train_steps, **kargs)
				learning_rate = self.warm_up(learning_rate, init_lr, **kargs)

			tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
			opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

			if kargs.get("use_tpu", 0) == 1:
				tf.logging.info("***** Using tpu cross shard optimizer *****")
				opt = tf.contrib.tpu.CrossShardOptimizer(opt)
			optimizer_dict[key] = opt

		for key in loss_dict:
			loss = loss_dict[key]
			tvars = tvars_dict[key]
			optimizer = optimizer_dict[key]
			grads = self.grad_clip_fn(loss, tvars, **kargs)
			with tf.variable_scope(key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
				train_op = optimizer.apply_gradients(
						zip(grads, tvars))
			opt_list.append(train_op)

		with tf.control_dependencies(opt_list):
			train_op = self.global_step.assign_add(1)
		return train_op

	def get_alternate_train_op(self, loss_dict, tvars_dict, init_lr_dict,
								optimizer_type_dict,
								num_train_steps, **kargs):
		prev_op = tf.no_op()

		loop_step_dict = kargs.get('loop_step_dict', None)
		if not loop_step_dict:
			loop_step_dict = {}
			for key in loss_dict:
				loop_step_dict[key] = 1

		if_grad_clip_dict = kargs.get('if_grad_clip_dict', None)
		if not if_grad_clip_dict:
			if_grad_clip_dict = {}
			for key in loss_dict:
				if_grad_clip_dict[key] = True

		optimizer_dict = {}

		alternate_order = kargs.get('alternate_order', list(loss_dict.keys()))
		print("==alternate order==", alternate_order)

		for key in alternate_order:
			init_lr = init_lr_dict[key]
			optimizer_type = optimizer_type_dict[key]
			if optimizer_type != 'radam':
				learning_rate = self.lr_decay_fn(init_lr, num_train_steps, **kargs)
				learning_rate = self.warm_up(learning_rate, init_lr, **kargs)

			tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
			opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

			if kargs.get("use_tpu", 0) == 1:
				tf.logging.info("***** Using tpu cross shard optimizer *****")
				opt = tf.contrib.tpu.CrossShardOptimizer(opt)
			optimizer_dict[key] = opt

		for key in alternate_order:
			loss = loss_dict[key]
			tvars = tvars_dict[key]
			loop_steps = loop_step_dict[key]
			optimizer = optimizer_dict[key]
			if_grad_clip = if_grad_clip_dict[key]

			# grads = self.grad_clip_fn(loss, tvars, **kargs)

			if if_grad_clip:
				grads = self.grad_clip_fn(loss, tvars, **kargs)
				tf.logging.info("==appy grad clip : %s==", key)
			else:
				grads = tf.gradients(loss, tvars)
				# grad_name = key
				# for grad, var in grads_and_vars:
				# 	if grad is not None:
				# 		var_grad_norm = tf.global_norm([grad])
				# 		tf.summary.scalar(grad_name+"/"+var.name, var_grad_norm)
				tf.logging.info("==not appy grad clip : %s==", key)

			for i in range(loop_steps):
				with tf.control_dependencies([prev_op]):
					with tf.variable_scope(key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
						prev_op = optimizer.apply_gradients(
							zip(grads, tvars))
						tf.logging.info("***** model: %s, step: %s *****", key, str(i))
		with tf.control_dependencies([prev_op]):
			train_op = self.global_step.assign_add(1)

		return train_op

	# def get_adaptive_alternate_train_op(self, loss_dict, tvars_dict, init_lr_dict,
	# 							optimizer_type_dict,
	# 							num_train_steps, **kargs):

	# 	loop_step_dict = kargs.get('loop_step_dict', None)
	# 	if not loop_step_dict:
	# 		loop_step_dict = {}
	# 		for key in loss_dict:
	# 			loop_step_dict[key] = 1

	# 	optimizer_dict = {}
	# 	global_step_dict = kargs.get('global_step_dict', None)
	# 	fce_acc = kargs.get("fce_acc", None)

	# 	for key in init_lr_dict:
	# 		init_lr = init_lr_dict[key]
	# 		optimizer_type = optimizer_type_dict[key]
	# 		if optimizer_type != 'radam':
	# 			learning_rate = self.private_lr_decay_fn(init_lr, num_train_steps,
	# 													global_step_dict[key], **kargs)
	# 			learning_rate = self.private_warm_up(learning_rate, init_lr, 
	# 												global_step_dict[key], **kargs)

	# 		tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
	# 		opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

	# 		if kargs.get("use_tpu", 0) == 1:
	# 			tf.logging.info("***** Using tpu cross shard optimizer *****")
	# 			opt = tf.contrib.tpu.CrossShardOptimizer(opt)
	# 		optimizer_dict[key] = opt

	# 	switch_acc = tf.get_variable(
	# 						"switch_acc",
	# 						shape=[],
	# 						initializer=tf.constant_initializer(0.0, dtype=tf.float32),
	# 						trainable=False)

	# 	postive_key = kargs.get("postive_key", "ebm")
	# 	negative_key = kargs.get("negative_key", "noise")

	# 	prev_ebm_op = tf.no_op()
	# 	with tf.control_dependencies([prev_ebm_op]): 
	# 		loss = loss_dict[postive_key]
	# 		tvars = tvars_dict[postive_key]
	# 		loop_steps = loop_step_dict[postive_key]
	# 		optimizer = optimizer_dict[postive_key]
	# 		loop_steps = loop_step_dict[postive_key]
	# 		grads = self.grad_clip_fn(optimizer, loss, tvars, grad_namepostive_key, **kargs)
	# 		for i in range(loop_steps):
	# 			with tf.control_dependencies([prev_ebm_op]):
	# 				with tf.variable_scope(postive_key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
	# 					prev_ebm_op = optimizer.apply_gradients(
	# 								grads_and_vars)
	# 					tf.logging.info("***** model: %s, step: %s *****", postive_key, str(i))
	# 		with tf.control_dependencies([prev_ebm_op]): 
	# 			prev_ebm_op = global_step_dict[postive_key].assign_add(1)

	# 	prev_noise_op = tf.no_op()
	# 	with tf.control_dependencies([prev_noise_op]): 
	# 		loss = loss_dict[negative_key]
	# 		tvars = tvars_dict[negative_key]
	# 		loop_steps = loop_step_dict[negative_key]
	# 		optimizer = optimizer_dict[negative_key]
	# 		loop_steps = loop_step_dict[negative_key]
	# 		grads = self.grad_clip_fn(optimizer, loss, tvars, grad_name=negative_key, **kargs)
	# 		for i in range(loop_steps):
	# 			with tf.control_dependencies([prev_noise_op]):
	# 				with tf.variable_scope(negative_key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
	# 					prev_noise_op = optimizer.apply_gradients(
	# 								grads_and_vars)
	# 					tf.logging.info("***** model: %s, step: %s *****", negative_key, str(i))
	# 		with tf.control_dependencies([prev_noise_op]): 
	# 			prev_noise_op = global_step_dict[negative_key].assign_add(1)

	# 	if kargs.get("use_tpu", 0) == 0:
	# 		tf.summary.scalar(postive_key+'_global_step', 
	# 							tf.reduce_sum(global_step_dict[postive_key]))
	# 		tf.summary.scalar(negative_key+'_global_step', 
	# 							tf.reduce_sum(global_step_dict[negative_key]))
	# 		tf.summary.scalar('switch_acc', 
	# 							tf.reduce_sum(switch_acc))

	# 	prev_op = tf.cond(tf.less_equal(switch_acc, 0.5),
	# 					   lambda: prev_ebm_op,
	# 					   lambda: prev_noise_op)
		
	# 	with tf.control_dependencies([prev_op]):
	# 		train_op = tf.group(switch_acc.assign(fce_acc), self.global_step.assign_add(1))
	# 	return train_op

