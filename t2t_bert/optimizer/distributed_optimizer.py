from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from optimizer import optimizer_utils
from optimizer import adam_weight_decay_utils

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

# pai_soar optimizer
try:
	import tensorflow.contrib.pai_soar as pai
except Exception as e:
	pai = None

try:
	import horovod.tensorflow as hvd
except Exception as e:
	hvd = None

class Optimizer(object):
	def __init__(self, config, **kargs):
		self.config = config
		self.global_step = tf.train.get_or_create_global_step()

		self.decay_global_step = tf.cond(tf.cast(self.global_step, tf.int64) < tf.constant(self.config.num_warmup_steps, dtype=tf.int64),
									lambda:tf.cast(tf.constant(0), tf.int64),
									lambda:self.global_step-tf.constant(self.config.num_warmup_steps, dtype=tf.int64))

	def lr_decay_fn(self, init_lr, num_train_steps,
					**kargs):
		lr_decay = self.config.get("lr_decay", "polynomial_decay")
		tf.logging.info(" lr decay method {}".format(lr_decay))
		learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
		end_learning_rate = self.config.get("end_learning_rate", 0.0)
		if lr_decay == "polynomial_decay":
			learning_rate = tf.train.polynomial_decay(
													learning_rate,
													self.decay_global_step,
													num_train_steps,
													end_learning_rate=end_learning_rate,
													power=1.0,
													cycle=False)
		elif lr_decay == "cosine_decay":
			learning_rate = tf.train.cosin_decay(
												learning_rate,
												self.decay_global_step,
												num_train_steps,
												alpha=0.0,
												cycle=False)
		elif lr_decay == "exponential_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.exponential_decay(
													learning_rate,
													self.decay_global_step,
													num_train_steps,
													decay_rate=decay_rate,
													staircase=False)
		elif lr_decay == "natural_exp_decay":
			decay_rate = self.config.get("lr_decay_rate", 0.999)
			learning_rate = tf.train.natural_exp_decay(
													learning_rate,
													self.decay_global_step,
													num_train_steps,
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

	def grad_clip_fn(self, opt, loss, tvars, **kargs):
		grads_and_vars = opt.compute_gradients(loss, tvars)
		grads = [grad for grad, _ in grads_and_vars]
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
		opt_type = self.config.get("train_op", "adam_decay")
		tf.logging.info(" optimization method {}".format(opt_type))
		if opt_type not in ["adam_decay", "adam"]:
			raise NotImplementedError()
		if opt_type == "adam_decay":
			opt = optimizer_utils.AdamWeightDecayOptimizer(
						learning_rate=learning_rate,
						weight_decay_rate=self.config.get("opt_decay_rate", 0.01),
						beta_1=self.config.get("beta_1", 0.9),
						beta_2=self.config.get("beta_2", 0.999),
						epsilon=self.config.get("epsilon", 1e-6),
						exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
		elif opt_type == "adam_weight_decay":
			opt = adam_weight_decay_utils.AdamWOptimizer(
						weight_decay=self.config.get("opt_decay_rate", 0.01),
						learning_rate=learning_rate,
						beta1=self.config.get("beta_1", 0.9),
						beta2=self.config.get("beta_2", 0.999),
               			epsilon=self.config.get("epsilon", 1e-6))
		elif opt_type == "adam":
			opt = tf.train.AdamOptimizer(learning_rate,
										beta1=self.config.get("beta_1", 0.9),
										beta2=self.config.get("beta_2", 0.999),
										epsilon=self.config.get("epsilon", 1e-8))
		return opt

	def get_opt(self, init_lr, 
				num_train_steps, **kargs):
		learning_rate = self.lr_decay_fn(init_lr, num_train_steps, **kargs)
		learning_rate = self.warm_up(learning_rate, init_lr, **kargs)
		
		# add uber horvod distributed optimizer
		if hvd and self.config["opt_type"] == "hvd":
			print("==optimizer hvd size=={}".format(hvd.size()))
			opt = self.optimizer_op(learning_rate*hvd.size(), **kargs)
			self.opt = hvd.DistributedOptimizer(opt)
		# add pai soar distributed optimizer
		elif pai and self.config["opt_type"] == "pai_soar":
			print("==optimizer pai_soar size=={}".format(self.config.get("worker_count", 4)))
			opt = self.optimizer_op(learning_rate*self.config.get("worker_count", 4), **kargs)
			self.opt = pai.ReplicatedVarsOptimizer(opt)
		# add tensorflow ps sync distributed optimizer
		elif self.config["opt_type"] == "ps_sync":
			print("==optimizer ps_sync size=={}".format(self.config.get("worker_count", 4)))
			opt = self.optimizer_op(learning_rate*self.config.get("worker_count", 4), **kargs)
			self.opt = tf.train.SyncReplicasOptimizer(opt, 
											replicas_to_aggregate=self.config.get("worker_count", 4), 
											total_num_replicas=self.config.get("worker_count", 4))
		else:
			print("==initialization of single node optimizer==")
			self.opt = self.optimizer_op(learning_rate, **kargs)

	def get_train_op(self, loss, tvars, init_lr, num_train_steps, **kargs):

		self.get_opt(init_lr, num_train_steps)

		grads = self.grad_clip_fn(self.opt, loss, tvars, **kargs)

		train_op = self.opt.apply_gradients(
					zip(grads, tvars), global_step=self.global_step)
		new_global_step = self.global_step + 1
		train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])
		return train_op