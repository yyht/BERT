import tensorflow as tf
import numpy as np
import re

try:
	from .trf_gpt_noise import model_fn_builder as noise_dist
	from .trf_ebm_bert import model_fn_builder as ebm_dist
	from .trf_classifier import get_ebm_loss, get_noise_loss, ebm_noise_train_metric, ebm_noise_eval_metric
except:
	from trf_gpt_noise import model_fn_builder as noise_dist
	from trf_ebm_bert import model_fn_builder as ebm_dist
	from trf_classifier import get_ebm_loss, get_noise_loss, ebm_noise_train_metric, ebm_noise_eval_metric

import tensorflow as tf
import numpy as np
from optimizer import optimizer
from optimizer import distributed_optimizer

from model_io import model_io

import tensorflow as tf
from metric import tf_metrics
from collections import OrderedDict

def get_train_op(optimizer_fn, opt_config,
				ebm_dist_config, noise_dist_config, 
				features, labels, mode, params,
				model_cls,
				**kargs):
	
	init_lr_dict = OrderedDict(zip(['ebm', 'noise'], [ebm_dist_config['init_lr'], noise_dist_config['init_lr']]))
	optimizer_type_dict = OrderedDict(zip(['ebm', 'noise'], [ebm_dist_config['optimizer_type'], noise_dist_config['optimizer_type']]))
	loop_step_dict = OrderedDict(zip(['ebm', 'noise'], [ebm_dist_config.get("steps", 1), noise_dist_config.get('steps', 1)]))
	
	switch_acc = tf.get_variable(
							"switch_acc",
							shape=[],
							initializer=tf.constant_initializer(0.0, dtype=tf.float32),
							trainable=False)

	postive_key = kargs.get("postive_key", "ebm")
	negative_key = kargs.get("negative_key", "noise")

	model_cls.get_opt(optimizer_fn, **kargs)
		
	def get_train_op(optimizer, loss, tvars, grad_name):
		grads_and_vars = optimizer_fn.grad_clip_fn(optimizer, loss, tvars, grad_name=postive_key, **kargs)
		with tf.variable_scope(grad_name+"/"+"optimizer", reuse=tf.AUTO_REUSE):
			op = optimizer.apply_gradients(
								grads_and_vars)
		return op

	prev_op = tf.no_op()
	scalar_loss = 0
	with tf.control_dependencies([prev_op]):
		model_cls.get_loss(features, labels, mode, params, **kargs)
		metric_dict = model_cls.metric_dict
		scalar_loss = model_cls.ebm_loss
		init_checkpoint = model_cls.var_checkpoint_dict_list

		def ebm_op():
			loss = model_cls.ebm_loss
			tvars = model_cls.ebm_vars
			opt = model_cls.opt_dict['ebm']
			prev_ebm_op = get_train_op(opt, loss, tvars, "ebm")
			return prev_ebm_op

		def noise_op():
			loss = model_cls.noise_loss
			tvars = model_cls.noise_vars
			opt = model_cls.opt['noise']
			prev_noise_op = get_train_op(opt, loss, tvars, "noise")
			return prev_noise_op

		prev_op = tf.cond(tf.less(switch_acc, 0.5),
						   ebm_op,
						   noise_op)

		with tf.control_dependencies([prev_op]):
			train_op = tf.group(switch_acc.assign(metric_dict['all_accuracy']), 
								optimizer_fn.global_step.assign_add(1))

			# prev_ebm_op = model_cls.global_step_dict['ebm'].assign_add(1)
			# prev_noise_op = model_cls.global_step_dict['noise'].assign_add(1)

	return train_op, scalar_loss, init_checkpoint

class EBM_NOISE_FCE(object):
	def __init__(self, model_config_dict,
						num_labels_dict,
						init_checkpoint_dict,
						load_pretrained_dict,
						model_io_config={},
						opt_config={},
						exclude_scope_dict={},
						not_storage_params_dict={},
						target_dict={},
						**kargs):
		self.model_config_dict = model_config_dict
		self.init_checkpoint_dict = init_checkpoint_dict
		self.load_pretrained_dict = load_pretrained_dict
		self.exclude_scope_dict = exclude_scope_dict
		self.target_dict = target_dict
		self.not_storage_params_dict = not_storage_params_dict
		self.model_io_config = model_io_config
		self.opt_config = opt_config
		self.num_labels_dict = num_labels_dict

		train_op_type = kargs.get('train_op_type', 'joint')

		self.ebm_dist_fn = ebm_dist(self.model_config_dict['ebm_dist'],
					self.num_labels_dict['ebm_dist'],
					self.init_checkpoint_dict['ebm_dist'],
					model_reuse=None,
					load_pretrained=self.load_pretrained_dict['ebm_dist'],
					model_io_config=self.model_io_config,
					opt_config=self.opt_config,
					exclude_scope=self.exclude_scope_dict.get('ebm_dist', ""),
					not_storage_params=self.not_storage_params_dict.get('ebm_dist', []),
					target=self.target_dict['ebm_dist'],
					**kargs)

		self.noise_dist_fn = noise_dist(self.model_config_dict['noise_dist'],
					self.num_labels_dict['noise_dist'],
					self.init_checkpoint_dict['noise_dist'],
					model_reuse=None,
					load_pretrained=self.load_pretrained_dict['noise_dist'],
					model_io_config=self.model_io_config,
					opt_config=self.opt_config,
					exclude_scope=self.exclude_scope_dict.get('noise_dist', ""),
					not_storage_params=self.not_storage_params_dict.get('noise_dist', []),
					target=self.target_dict['noise_dist'],
					noise_true_distribution=True,
					sample_noise_dist=True,
					noise_estimator_type=kargs.get("noise_estimator_type", "stop_gradient"),
					**kargs)

	def get_opt(self, optimizer_fn, **kargs):

		num_train_steps = self.opt_config.num_train_steps

		self.init_lr_dict = OrderedDict(zip(['ebm', 'noise'], [self.model_config_dict['ebm_dist']['init_lr'], self.model_config_dict['noise_dist']['init_lr']]))
		self.optimizer_type_dict = OrderedDict(zip(['ebm', 'noise'], [self.model_config_dict['ebm_dist']['optimizer_type'], self.model_config_dict['noise_dist']['optimizer_type']]))
		self.loop_step_dict = OrderedDict(zip(['ebm', 'noise'], [self.model_config_dict['ebm_dist'].get("steps", 1), self.model_config_dict['noise_dist'].get('steps', 1)]))

		# logits is logp, when we need to directly maximize it, we only minus
		with tf.variable_scope("ebm", reuse=tf.AUTO_REUSE):
			self.ebm_global_step = tf.get_variable(
								"global_step",
								shape=[],
								initializer=tf.constant_initializer(0, dtype=tf.int64),
								trainable=False,
								dtype=tf.int64)

		with tf.variable_scope("noise", reuse=tf.AUTO_REUSE):
			self.noise_global_step = tf.get_variable(
								"global_step",
								shape=[],
								initializer=tf.constant_initializer(0, dtype=tf.int64),
								trainable=False,
								dtype=tf.int64)

		self.global_step_dict = {
			"ebm":self.ebm_global_step,
			"noise":self.noise_global_step
		}

		self.opt_dict = {}
		for key in ['ebm', 'noise']:

			init_lr = self.init_lr_dict[key]
			optimizer_type = self.optimizer_type_dict[key]
			if optimizer_type != 'radam':
				learning_rate = optimizer_fn.private_lr_decay_fn(init_lr, num_train_steps,
														self.global_step_dict[key], **kargs)
				learning_rate = optimizer_fn.private_warm_up(learning_rate, init_lr, 
														self.global_step_dict[key], **kargs)

			tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
			opt = optimizer_fn.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

			if kargs.get("use_tpu", 0) == 1:
				tf.logging.info("***** Using tpu cross shard optimizer *****")
				opt = tf.contrib.tpu.CrossShardOptimizer(opt)
			self.opt_dict[key] = opt

	def get_loss(self, features, labels, mode, params, **kargs):
		ebm_true_features = {}
		noise_true_features = {}
		for key in features:
			if key == 'input_ori_ids':
				ebm_true_features["input_ids"] = tf.identity(features['input_ori_ids'])
				noise_true_features["input_ids"] = tf.identity(features['input_ori_ids'])
			if key in ['input_mask', 'segment_ids']:
				ebm_true_features[key] = tf.identity(features[key])
				noise_true_features[key] = tf.identity(features[key])

		# first get noise dict
		self.noise_dist_dict = self.noise_dist_fn(noise_true_features, labels, mode, params)

		# second, get true ebm dict
		self.true_ebm_dist_dict = self.ebm_dist_fn(ebm_true_features, labels, mode, params)

		# third, get fake ebm dict
		ebm_fake_features = {}
		for key in features:
			if key in ['input_mask', 'segment_ids']:
				ebm_fake_features[key] = tf.identity(features[key])

		if kargs.get("training_mode", "stop_gradient") == 'stop_gradient':
			ebm_fake_features["input_ids"] = tf.identity(self.noise_dist_dict['fake_samples'])
		elif kargs.get("training_mode", "adv_gumbel") == 'adv_gumbel':
			ebm_fake_features["input_ids"] = tf.identity(self.noise_dist_dict['gumbel_probs'])

		self.fake_ebm_dist_dict = self.ebm_dist_fn(ebm_fake_features, labels, mode, params)

		# log(1/(1+exp(logp_n-logp_ebm)))
		# for sigmoid simplicity, we just minus 

		ebm_loss_ratio = kargs.get('ebm_loss_ratio', 1.0)
		noise_loss_ratio = kargs.get('noise_loss_ratio', 1.0)

		self.ebm_loss = get_ebm_loss(self.true_ebm_dist_dict['logits'], 
								self.noise_dist_dict['true_logits'], 
								self.fake_ebm_dist_dict['logits'], 
								self.noise_dist_dict['fake_logits'], 
								use_tpu=kargs.get('use_tpu', False))

		self.ebm_loss *= ebm_loss_ratio

		self.noise_loss = get_noise_loss(self.true_ebm_dist_dict['logits'], 
									self.noise_dist_dict['true_logits'], 
									self.fake_ebm_dist_dict['logits'], 
									self.noise_dist_dict['fake_logits'], 
									noise_loss_type=kargs.get('noise_loss_type', 'jsd_noise'))
		
		self.noise_loss *= noise_loss_ratio

		self.ebm_vars = self.true_ebm_dist_dict['tvars']
		self.noise_vars = self.noise_dist_dict['tvars']

		self.metric_dict = ebm_noise_train_metric(
										self.true_ebm_dist_dict['logits'], 
										self.noise_dist_dict['true_logits'], 
										self.fake_ebm_dist_dict['logits'], 
										self.noise_dist_dict['fake_logits'],
										features['input_ori_ids'],
										tf.cast(features['input_mask'], tf.float32),
										self.noise_dist_dict["true_seq_logits"],
										)

		if not kargs.get('use_tpu', False) and kargs.get('if_summary', True):
			for key in self.metric_dict:
				tf.summary.scalar(key, self.metric_dict[key])
			tf.summary.scalar("ebm_loss", self.ebm_loss)
			tf.summary.scalar("noise_loss", self.noise_loss)

		self.var_checkpoint_dict_list = []
		for key in self.init_checkpoint_dict:
			if self.load_pretrained_dict[key] == "yes":
				if key == 'ebm_dist':
					tmp = {
							"tvars":self.ebm_vars,
							"init_checkpoint":self.init_checkpoint_dict['ebm_dist'],
							"exclude_scope":self.exclude_scope_dict[key],
							"restore_var_name":self.model_config_dict['ebm_dist'].get('restore_var_name', [])
					}
					if kargs.get("sharing_mode", "none") != "none":
						tmp['exclude_scope'] = ''
					self.var_checkpoint_dict_list.append(tmp)
				elif key == 'noise_dist':
					tmp = {
							"tvars":self.noise_vars,
							"init_checkpoint":self.init_checkpoint_dict['noise_dist'],
							"exclude_scope":self.exclude_scope_dict[key],
							"restore_var_name":self.model_config_dict['noise_dist'].get('restore_var_name', [])
					}
					self.var_checkpoint_dict_list.append(tmp)

def classifier_model_fn_builder(
						model_config_dict,
						num_labels_dict,
						init_checkpoint_dict,
						load_pretrained_dict,
						model_io_config={},
						opt_config={},
						exclude_scope_dict={},
						not_storage_params_dict={},
						target_dict={},
						**kargs):
	
	def model_fn(features, labels, mode, params):

		ebm_noise_fce = EBM_NOISE_FCE(model_config_dict,
									num_labels_dict,
									init_checkpoint_dict,
									load_pretrained_dict,
									model_io_config=model_io_config,
									opt_config=opt_config,
									exclude_scope_dict=exclude_scope_dict,
									not_storage_params_dict=not_storage_params_dict,
									target_dict=target_dict,
									**kargs)

		model_io_fn = model_io.ModelIO(model_io_config)

		if mode == tf.estimator.ModeKeys.TRAIN:

			if kargs.get('use_tpu', False):
				optimizer_fn = optimizer.Optimizer(opt_config)
				use_tpu = 1
			else:
				optimizer_fn = distributed_optimizer.Optimizer(opt_config)
				use_tpu = 0

			train_op, loss, var_checkpoint_dict_list = get_train_op(
								optimizer_fn, opt_config,
								model_config_dict['ebm_dist'], 
								model_config_dict['noise_dist'],
								features, labels, mode, params,
								ebm_noise_fce,
								use_tpu=use_tpu)

			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			
			if len(var_checkpoint_dict_list) >= 1:
				scaffold_fn = model_io_fn.load_multi_pretrained(
												var_checkpoint_dict_list,
												use_tpu=use_tpu)
			else:
				scaffold_fn = None

			if kargs.get('use_tpu', False):
				estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
								mode=mode,
								loss=loss,
								train_op=train_op,
								scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(
								mode=mode, 
								loss=loss, 
								train_op=train_op)

			return estimator_spec

		elif mode == tf.estimator.ModeKeys.EVAL:

			ebm_noise_fce.get_loss(features, labels, mode, params, **kargs)

			tpu_eval_metrics = (ebm_noise_eval_metric, 
								[
								ebm_noise_fce.true_ebm_dist_dict['logits'], 
								ebm_noise_fce.noise_dist_dict['true_logits'], 
								ebm_noise_fce.fake_ebm_dist_dict['logits'], 
								ebm_noise_fce.noise_dist_dict['fake_logits'],
								features['input_ori_ids'],
								tf.cast(features['input_mask'], tf.float32),
								ebm_noise_fce.noise_dist_dict["true_seq_logits"]
								])
			gpu_eval_metrics = ebm_noise_eval_metric(
								ebm_noise_fce.true_ebm_dist_dict['logits'], 
								ebm_noise_fce.noise_dist_dict['true_logits'], 
								ebm_noise_fce.fake_ebm_dist_dict['logits'], 
								ebm_noise_fce.noise_dist_dict['fake_logits'],
								features['input_ori_ids'],
								tf.cast(features['input_mask'], tf.float32),
								ebm_noise_fce.noise_dist_dict["true_seq_logits"]
								)

			loss = ebm_noise_fce.ebm_loss + ebm_noise_fce.noise_loss
			var_checkpoint_dict_list = ebm_noise_fce.var_checkpoint_dict_list

			if len(var_checkpoint_dict_list) >= 1:
				scaffold_fn = model_io_fn.load_multi_pretrained(
												var_checkpoint_dict_list,
												use_tpu=use_tpu)
			else:
				scaffold_fn = None

			if kargs.get('use_tpu', False):
				estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
							  mode=mode,
							  loss=loss,
							  eval_metrics=tpu_eval_metrics,
							  scaffold_fn=scaffold_fn)
			else:
				estimator_spec = tf.estimator.EstimatorSpec(mode=mode, 
								loss=loss,
								eval_metric_ops=gpu_eval_metrics)

			return estimator_spec
		else:
			raise NotImplementedError()

	return model_fn


