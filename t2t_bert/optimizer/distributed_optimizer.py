from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from optimizer import optimizer_utils
from optimizer import adam_weight_decay_utils
from optimizer import adam_weight_decay_exclude_utils
from optimizer import pai_soar_optimizer_utils
from optimizer import radam_utils

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np

# pai_soar optimizer
try:
	import paisoar as pai
except Exception as e:
	pai = None

try:
	import horovod.tensorflow as hvd
except Exception as e:
	hvd = None

class Optimizer(object):
	def __init__(self, config, **kargs):
		self.config = config

		# self.global_step = tf.get_variable(
		#                   "global_step",
		#                   dtype=tf.int64,
		#                   initializer=tf.constant(1, dtype=tf.int64),
		#                   reuse=tf.AUTO_REUSE)

		self.global_step = tf.train.get_or_create_global_step()

		cond_fn = tf.less(self.global_step, tf.constant(self.config.num_warmup_steps, dtype=tf.int64))

		self.decay_global_step = tf.cond(cond_fn,
									lambda:tf.constant(value=0, shape=[], dtype=tf.int64, name="initial_global_step"),
									lambda:self.global_step-tf.constant(self.config.num_warmup_steps, dtype=tf.int64))

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

	def lr_decay_fn(self, init_lr, num_train_steps,
					**kargs):
		lr_decay = self.config.get("lr_decay", "polynomial_decay")
		tf.logging.info(" lr decay method {}".format(lr_decay))
		learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32, name="init_lr")
		end_learning_rate = self.config.get("end_learning_rate", 0.0)
		if lr_decay == "polynomial_decay":
			learning_rate = tf.train.polynomial_decay(
													init_lr,
													self.decay_global_step,
													num_train_steps-self.config.num_warmup_steps,
													end_learning_rate=end_learning_rate,
													power=1.0,
													cycle=False)
		elif lr_decay == "cosine_decay":
			learning_rate = tf.train.cosin_decay(
												learning_rate,
												self.decay_global_step,
												num_train_steps-self.config.num_warmup_steps,
												alpha=0.0,
												cycle=False)
		elif lr_decay == "exponential_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.exponential_decay(
													learning_rate,
													self.decay_global_step,
													num_train_steps-self.config.num_warmup_steps,
													decay_rate=decay_rate,
													staircase=False)
		elif lr_decay == "natural_exp_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.natural_exp_decay(
													learning_rate,
													self.decay_global_step,
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

	def grad_clip_fn(self, opt, loss, tvars, **kargs):
		gpu_count = self.config.get('gpu_count', 1)
		grad_name = kargs.get('grad_name', "grad_norm")
		if self.config.get("opt_type", "pai_soar") == "pai_soar":
			loss_fn = opt.compute_loss(loss, loss_scale=self.config.get("loss_scale", 1))
			grads_and_vars = opt.compute_gradients(loss_fn, colocate_gradients_with_ops=True)
		else:   
			grads_and_vars = opt.compute_gradients(loss, tvars)

			valid_vars = []
			for grad, var in grads_and_vars:
				if grad is not None:
					valid_vars.append(var)
				else:
					print(var.name, "=====none grad======", grad_name)

			grads = [grad/gpu_count for grad, _ in grads_and_vars if grad is not None] # allreduce from sum to mean
			# grads_and_vars = zip(valid_grads, valid_vars)
			grad_clip = self.config.get("grad_clip", "global_norm")
			use_norm = tf.global_norm(grads)
			tf.summary.scalar(grad_name+'/total_grad_norm', use_norm)
			for grad, var in grads_and_vars:
				if grad is not None:
					var_grad_norm = tf.global_norm([grad])
					tf.summary.scalar(grad_name+"/"+var.name, var_grad_norm)
				# tf.summary.histogram(var.name, var)
				# tf.summary.histogram("grad/"+var.name, grad)

			tf.logging.info(" gradient clip method {}".format(grad_clip))
			
			if grad_clip == "global_norm":
				clip_norm = self.config.get("clip_norm", 1.0)
				if self.config.get("strategy", "") in ['MirroredStrategy', 'CollectiveAllReduceStrategy']:
					use_norm = tf.global_norm(grads)

					[scale_grads, _] = tf.clip_by_global_norm(grads, 
										clip_norm=clip_norm,
										use_norm=use_norm*tf.sqrt(gpu_count*1.0))

					tf.summary.scalar(grad_name+'/grad_scale', use_norm*tf.sqrt(gpu_count*1.0))
				else:
					[scale_grads, _] = tf.clip_by_global_norm(grads, 
										clip_norm=clip_norm)
			elif grad_clip == "norm":
				clip_norm = self.config.get("clip_norm", 1.0)
				scale_grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
			elif grad_clip == "value":
				clip_min_value = self.config.get("clip_min_value", -1.0)
				clip_max_value = self.config.get("clip_max_value", 1.0)
				scale_grads = [tf.clip_by_value(grad, clip_norm) for grad in grads]
			else:
				scale_grads = grads
			
			grads_and_vars = zip(scale_grads, valid_vars)

		return grads_and_vars

	def optimizer_op(self, learning_rate,
							**kargs):
		opt_type = self.config.get("train_op", "adam_decay")
		tf.logging.info(" optimization method {}".format(opt_type))
		if opt_type not in ["adam_decay", "adam", "adam_weight_decay", 
					"adam_weight_decay_exclude", "pai_soar_adam_decay", "lamb",
					"adafactor"]:
			raise NotImplementedError()
		if opt_type == "adam_decay":
			print("==apply bert adam weight decay==")
			opt = optimizer_utils.AdamWeightDecayOptimizer(
						learning_rate=learning_rate,
						weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
						beta_1=self.config.get("beta_1", 0.9),
						beta_2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6),
						exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
		elif opt_type == "adam_weight_decay":
			print("==apply original adam weight decay==")
			opt = adam_weight_decay_utils.AdamWOptimizer(
						weight_decay=self.config.get("opt_decay_rate", 0.01),
						learning_rate=learning_rate,
						beta1=self.config.get("beta_1", 0.9),
						beta2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6))
		elif opt_type == "adam_weight_decay_exclude":
			print("==apply adam weight decay==")
			opt = adam_weight_decay_exclude_utils.AdamWOptimizer(
						weight_decay=self.config.get("opt_decay_rate", 0.01),
						learning_rate=learning_rate,
						beta1=self.config.get("beta_1", 0.9),
						beta2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6),
						exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
						)
		elif opt_type == "pai_soar_adam_decay":
			print("==apply pai soar adam decay==")
			opt = pai_soar_optimizer_utils.AdamWeightDecayOptimizer(
						learning_rate=learning_rate,
						weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
						beta_1=self.config.get("beta_1", 0.9),
						beta_2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6),
						exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
		elif opt_type == "adam":
			print("==apply adam==")
			opt = tf.train.AdamOptimizer(learning_rate,
										beta1=self.config.get("beta_1", 0.9),
										beta2=self.config.get("beta_2", 0.999),
										epsilon=self.config.get("epsilon", 1e-6))
		elif opt_type == 'lamb':
			print("==apply lamb==")
			opt = optimizer_utils.LAMBOptimizer(
								learning_rate,
								 weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
								 beta_1=self.config.get("beta_1", 0.9),
								 beta_2=self.config.get("beta_2", 0.999),
								 epsilon=self.config.get("epsilon", 1e-6),
								 exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
								)
		elif opt_type == 'radam':
			opt = radam_utils.RAdamOptimizer(
								learning_rate,
								 weight_decay=self.config.get("opt_decay_rate", 0.0),
								 beta1=self.config.get("beta_1", 0.9),
								 beta2=self.config.get("beta_2", 0.999),
								 epsilon=self.config.get("epsilon", 1e-6)
								)
		elif opt_type == "adafactor":
			print("=== apply adafactor ===")
			opt =  optimizer_utils.AdaFactorOptimizer(
								learning_rate=learning_rate,
								weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
								beta_1=self.config.get("beta_1", 0.9),
								beta_2=self.config.get("beta_2", 0.999),
								epsilon=self.config.get("epsilon", 1e-6),
								exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

		if self.config.get("opt_ema", "no") == "yes":
			print("==apply ema optimizer==")
			opt = tf.contrib.opt.MovingAverageOptimizer(opt)
		return opt

	def get_opt(self, init_lr, 
				num_train_steps, **kargs):

		learning_rate = init_lr
		if self.config.get("decay", "no") == "decay":
			print("==apply lr decay==")
			learning_rate = self.lr_decay_fn(learning_rate, num_train_steps, **kargs)
		if self.config.get("warmup", "no") == "warmup":
			print("==apply warmup==")
			learning_rate = self.warm_up(learning_rate, init_lr, **kargs)
		self.learning_rate = learning_rate #* (self.config.get('gpu_count', 1) / 2)
		# self.learning_rate = learning_rate / np.sqrt(self.config.get('gpu_count', 1) / 2)
		# self.learning_rate = learning_rate * np.sqrt(self.config.get('gpu_count', 1)) * 2
		self.single_node_learning = learning_rate
		
		# add uber horvod distributed optimizer
		if hvd and self.config["opt_type"] == "hvd":
			print("==optimizer hvd size=={}".format(self.config.get("worker_count", hvd.size())))
			opt = self.optimizer_op(self.learning_rate*self.config.get("worker_count", hvd.size()), **kargs)
			self.opt = hvd.DistributedOptimizer(opt)
			self.distributed_hooks = [hvd.BroadcastGlobalVariablesHook(0)]
		# add pai soar distributed optimizer
		elif pai and self.config["opt_type"] == "pai_soar":
			print("==optimizer pai_soar size=={}".format(self.config.get("worker_count", 4)))
			opt = self.optimizer_op(self.learning_rate*self.config.get("worker_count", 4), **kargs)
			self.opt = pai.ReplicatedVarsOptimizer(opt, clip_norm=self.config.get("clip_norm", 1.0))
			self.distributed_hooks = []
		# add tensorflow ps sync distributed optimizer
		elif self.config["opt_type"] == "ps_sync":
			print("==optimizer ps_sync size=={}".format(self.config.get("worker_count", 4)))
			opt = self.optimizer_op(self.learning_rate*self.config.get("worker_count", 4), **kargs)
			self.opt = tf.train.SyncReplicasOptimizer(opt, 
											replicas_to_aggregate=self.config.get("worker_count", 4), 
											total_num_replicas=self.config.get("worker_count", 4))
			self.distributed_hooks = [self.opt.make_session_run_hook(self.config["is_chief"], num_tokens=0)]
		elif self.config["opt_type"] == "ps":
			print("==optimizer ps_async size=={}".format(self.config.get("worker_count", 4)))
			self.opt = self.optimizer_op(self.learning_rate*self.config.get("worker_count", 4), **kargs)
		else:
			print("==initialization of single node optimizer==")
			self.opt = self.optimizer_op(self.learning_rate, **kargs)
			self.distributed_hooks = []

	def get_train_op(self, loss, tvars, init_lr, num_train_steps, **kargs):
		
		self.get_opt(init_lr, num_train_steps)

		grads_and_vars = self.grad_clip_fn(self.opt, loss, tvars, **kargs)

		train_op = self.opt.apply_gradients(
					grads_and_vars, global_step=self.global_step)
		new_global_step = self.global_step + 1
		if self.config.get("train_op", None) in ["adam_decay", "adafactor", "lamb"]:
			train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
		train_op = train_op
		return train_op

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
			grads_and_vars = self.grad_clip_fn(optimizer, loss, tvars, grad_name=key, **kargs)
			with tf.variable_scope(key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
				train_op = optimizer.apply_gradients(
						grads_and_vars)
			opt_list.append(train_op)

		with tf.control_dependencies(opt_list):
			train_op = self.global_step.assign_add(1)
		return train_op

	def gradient_norm_summary(self, loss, tvars, **kargs):
		local_opt = tf.train.AdamOptimizer(0.01,
										beta1=self.config.get("beta_1", 0.9),
										beta2=self.config.get("beta_2", 0.999),
										epsilon=self.config.get("epsilon", 1e-6))

		debug_grad_name = kargs.get('debug_grad_name', 'original')
		# for var in tvars:
		#   print(var, "=====debug_grad_name====", debug_grad_name)

		grads_and_vars = local_opt.compute_gradients(loss, tvars)
		
		local_grads = []
		local_vars = []

		for grad, var in grads_and_vars:
			if grad is not None:
				local_grads.append(grad)
				local_vars.append(var)
				continue
			else:
				print(var.name, "=====none grad======")

		grad_clip = self.config.get("grad_clip", "global_norm")
		use_norm = tf.global_norm(local_grads)
		tf.summary.scalar(debug_grad_name+'/total_grad_norm', use_norm)
		for grad, var in zip(local_grads, local_vars):
			var_grad_norm = tf.global_norm([grad])
			tf.summary.scalar(debug_grad_name+"/"+var.name, var_grad_norm)

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
		if not loop_step_dict:
			loop_step_dict = {}
			for key in loss_dict:
				loop_step_dict[key] = True

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

			print(key, "====apply gradients====", if_grad_clip)
			if if_grad_clip:
				grads_and_vars = self.grad_clip_fn(optimizer, loss, tvars, grad_name=key,
												**kargs)
				tf.logging.info("==appy grad clip : %s==", key)
			else:
				tf.logging.info("==not appy grad clip : %s==", key)
				print(tvars, "=======logz=======", key)
				grads_and_vars = optimizer.compute_gradients(loss, tvars)
				grad_name = key
				for grad, var in grads_and_vars:
					if grad is not None:
						var_grad_norm = tf.global_norm([grad])
						tf.summary.scalar(grad_name+"/"+var.name, var_grad_norm)
				
			for i in range(loop_steps):
				with tf.control_dependencies([prev_op]):
					with tf.variable_scope(key+"/"+"optimizer", reuse=tf.AUTO_REUSE):
						prev_op = optimizer.apply_gradients(
							grads_and_vars)
						tf.logging.info("***** model: %s, step: %s *****", key, str(i))
		with tf.control_dependencies([prev_op]):
			train_op = self.global_step.assign_add(1)

		return train_op

	# def get_adaptive_alternate_train_op(self, loss_dict, tvars_dict, init_lr_dict,
	# 							optimizer_type_dict,
	# 							num_train_steps, **kargs):

	# 	prev_op = tf.no_op()

	# 	loop_step_dict = kargs.get('loop_step_dict', None)
	# 	if not loop_step_dict:
	# 		loop_step_dict = {}
	# 		for key in loss_dict:
	# 			loop_step_dict[key] = 1

	# 	optimizer_dict = {}

	# 	global_step_dict = kargs.get('global_step_dict', None)
	# 	fce_acc = kargs.get("fce_acc", None)

	# 	# for key in init_lr_dict:
	# 	# 	init_lr = init_lr_dict[key]
	# 	# 	optimizer_type = optimizer_type_dict[key]
	# 	# 	if optimizer_type != 'radam':
	# 	# 		learning_rate = self.private_lr_decay_fn(init_lr, num_train_steps,
	# 	# 												global_step_dict[key], **kargs)
	# 	# 		learning_rate = self.private_warm_up(learning_rate, init_lr, 
	# 	# 											global_step_dict[key], **kargs)

	# 	# 	tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
	# 	# 	opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

	# 	# 	if kargs.get("use_tpu", 0) == 1:
	# 	# 		tf.logging.info("***** Using tpu cross shard optimizer *****")
	# 	# 		opt = tf.contrib.tpu.CrossShardOptimizer(opt)
	# 	# 	optimizer_dict[key] = opt

	# 	switch_acc = tf.get_variable(
	# 						"switch_acc",
	# 						shape=[],
	# 						initializer=tf.constant_initializer(0.0, dtype=tf.float32),
	# 						trainable=False)

	# 	postive_key = kargs.get("postive_key", "ebm")
	# 	negative_key = kargs.get("negative_key", "noise")

	# 	def get_train_op(optimizer, loss, tvars, grad_name):
	# 		grads_and_vars = self.grad_clip_fn(optimizer, loss, tvars, grad_name=postive_key, **kargs)
	# 		with tf.variable_scope(grad_name+"/"+"optimizer", reuse=tf.AUTO_REUSE):
	# 			op = optimizer.apply_gradients(
	# 								grads_and_vars)
	# 		return op
		
	# 	def ebm_op():
	# 		loop_steps = loop_step_dict[postive_key]

	# 		init_lr = init_lr_dict[postive_key]
	# 		optimizer_type = optimizer_type_dict[postive_key]
	# 		if optimizer_type != 'radam':
	# 			learning_rate = self.private_lr_decay_fn(init_lr, num_train_steps,
	# 													global_step_dict[postive_key], **kargs)
	# 			learning_rate = self.private_warm_up(learning_rate, init_lr, 
	# 												global_step_dict[postive_key], **kargs)

	# 		tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", postive_key, optimizer_type, str(init_lr))
	# 		opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

	# 		if kargs.get("use_tpu", 0) == 1:
	# 			tf.logging.info("***** Using tpu cross shard optimizer *****")
	# 			opt = tf.contrib.tpu.CrossShardOptimizer(opt)

	# 		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# 		with tf.control_dependencies(update_ops):
	# 			prev_ebm_op = tf.no_op()
	# 			for i in range(loop_steps):
	# 				with tf.control_dependencies([prev_ebm_op]):
	# 					tvars = tvars_dict[postive_key]
	# 					loop_steps = loop_step_dict[postive_key]
	# 					loss = loss_dict[postive_key]
	# 					prev_ebm_op = get_train_op(opt, loss, tvars, postive_key)
	# 					tf.logging.info("***** model: %s, step: %s *****", postive_key, str(i))
	# 			with tf.control_dependencies([prev_ebm_op]): 
	# 				prev_ebm_op = global_step_dict[postive_key].assign_add(1)
	# 		return prev_ebm_op

	# 	def noise_op():
	# 		loop_steps = loop_step_dict[negative_key]
	# 		init_lr = init_lr_dict[negative_key]
	# 		optimizer_type = optimizer_type_dict[negative_key]
	# 		if optimizer_type != 'radam':
	# 			learning_rate = self.private_lr_decay_fn(init_lr, num_train_steps,
	# 													global_step_dict[negative_key], **kargs)
	# 			learning_rate = self.private_warm_up(learning_rate, init_lr, 
	# 												global_step_dict[negative_key], **kargs)

	# 		tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", negative_key, optimizer_type, str(init_lr))
	# 		opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

	# 		if kargs.get("use_tpu", 0) == 1:
	# 			tf.logging.info("***** Using tpu cross shard optimizer *****")
	# 			opt = tf.contrib.tpu.CrossShardOptimizer(opt)

	# 		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	# 		with tf.control_dependencies(update_ops):
	# 			prev_noise_op = tf.no_op()
	# 			for i in range(loop_steps):
	# 				with tf.control_dependencies([prev_noise_op]):
	# 					loss = loss_dict[negative_key]
	# 					tvars = tvars_dict[negative_key]
	# 					loop_steps = loop_step_dict[negative_key]
	# 					prev_noise_op = get_train_op(opt, loss, tvars, negative_key)
	# 					tf.logging.info("***** model: %s, step: %s *****", negative_key, str(i))

	# 			with tf.control_dependencies([prev_noise_op]): 
	# 				prev_noise_op = global_step_dict[negative_key].assign_add(1)
	# 		return prev_noise_op

	# 	if kargs.get("use_tpu", 0) == 0:
	# 		tf.summary.scalar(postive_key+'_global_step', 
	# 							tf.reduce_sum(global_step_dict[postive_key]))
	# 		tf.summary.scalar(negative_key+'_global_step', 
	# 							tf.reduce_sum(global_step_dict[negative_key]))
	# 		tf.summary.scalar('switch_acc', 
	# 							tf.reduce_sum(switch_acc))

	# 	prev_op = tf.cond(tf.equal(tf.mod(self.global_step, 5), 0),
	# 					   ebm_op,
	# 					   noise_op)

	# 	train_op = tf.group(prev_op, self.global_step.assign_add(1), switch_acc.assign(fce_acc))

	# 	return train_op

	# def get_adaptive_alternate_train_op_v1(self, init_lr_dict,
	# 							optimizer_type_dict,
	# 							num_train_steps, 
	# 							features, 
	# 							labels, 
	# 							mode, 
	# 							params,
	# 							model_cls,
	# 							**kargs):

	# 	prev_op = tf.no_op()

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

	# 	def get_train_op(optimizer, loss, vars, grad_name):
	# 		grads_and_vars = self.grad_clip_fn(optimizer, loss, tvars, grad_name=postive_key, **kargs)
	# 		with tf.variable_scope(grad_name+"/"+"optimizer", reuse=tf.AUTO_REUSE):
	# 			op = optimizer.apply_gradients(
	# 								grads_and_vars)
	# 		return op

	# 	prev_ebm_op = tf.no_op()
	# 	loop_steps = loop_step_dict[postive_key]
	# 	for i in range(loop_steps):
	# 		with tf.control_dependencies([prev_ebm_op]):
	# 			model_cls.get_loss(features, labels, mode, params, if_summary=True, **kargs)
	# 			loss = tf.identity(model_cls.ebm_loss)
	# 			loop_steps = loop_step_dict[postive_key]
	# 			optimizer = optimizer_dict[postive_key]
				
	# 			prev_ebm_op = get_train_op(optimizer, loss, model_cls.ebm_vars, postive_key)
	# 			tf.logging.info("***** model: %s, step: %s *****", postive_key, str(i))

	# 	with tf.control_dependencies([prev_ebm_op]): 
	# 		prev_ebm_op = global_step_dict[postive_key].assign_add(1)

	# 	prev_noise_op = tf.no_op()
	# 	loop_steps = loop_step_dict[negative_key]
	# 	for i in range(loop_steps):
	# 		with tf.control_dependencies([prev_noise_op]):
	# 			model_cls.get_loss(features, labels, mode, params, if_summary=False, **kargs)
	# 			loss = tf.identity(model_cls.noise_loss)
	# 			loop_steps = loop_step_dict[negative_key]
	# 			optimizer = optimizer_dict[negative_key]
	# 			prev_noise_op = get_train_op(optimizer, loss, model_cls.noise_vars, negative_key)
	# 			tf.logging.info("***** model: %s, step: %s *****", negative_key, str(i))

	# 	with tf.control_dependencies([prev_noise_op]): 
	# 		prev_noise_op = global_step_dict[negative_key].assign_add(1)

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
	# 		train_op = tf.group(self.global_step.assign_add(1), switch_acc.assign(fce_acc))

	# 	return train_op



	