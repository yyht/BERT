import tensorflow as tf
import numpy as np
import re
from utils.bert import bert_utils
try:
	from .trf_gpt_noise import model_fn_builder as noise_dist
	from .trf_ebm_bert import model_fn_builder as ebm_dist
	from .trf_classifier import get_ebm_loss, get_residual_ebm_loss, get_ebm_mlm_adv_loss, get_noise_loss, ebm_noise_train_metric, ebm_noise_eval_metric, ebm_eval_metric, ebm_train_metric
	from .trf_ebm_noise_mlm_sample import model_fn_builder as mlm_noise_dist
except:
	from trf_gpt_noise import model_fn_builder as noise_dist
	from trf_ebm_bert import model_fn_builder as ebm_dist
	from trf_ebm_noise_mlm_sample import model_fn_builder as mlm_noise_dist
	from trf_classifier import get_ebm_loss, get_residual_ebm_loss, get_ebm_mlm_adv_loss, get_noise_loss, ebm_noise_train_metric, ebm_noise_eval_metric, ebm_eval_metric, ebm_train_metric

import tensorflow as tf
import numpy as np
from optimizer import optimizer
from optimizer import distributed_optimizer

from model_io import model_io

import tensorflow as tf
from metric import tf_metrics
from collections import OrderedDict

def get_train_op(model_cls, optimizer_fn, opt_config,
				ebm_dist_config, noise_dist_config,
				mlm_dist_config,
				features, labels, mode, params,
				**kargs):
	
	init_lr_dict = OrderedDict(zip(['ebm', 'ebm_logz', 'generator'], [ebm_dist_config['init_lr'], ebm_dist_config.get('logz_init_lr', ebm_dist_config['init_lr']), mlm_dist_config['init_lr']]))
	optimizer_type_dict = OrderedDict(zip(['ebm', 'ebm_logz', 'generator'], [ebm_dist_config['optimizer_type'], ebm_dist_config['logz_optimizer_type'], mlm_dist_config['optimizer_type']]))
	loop_step_dict = OrderedDict(zip(['ebm', 'ebm_logz', 'generator'], [ebm_dist_config.get("steps", 1), ebm_dist_config.get("logz_steps", 1), mlm_dist_config.get("steps", 1)]))
	if_grad_clip_dict = OrderedDict(zip(['ebm', 'ebm_logz', 'generator'], [True, True, True]))
	
	use_tpu = 1 if kargs.get('use_tpu', False) else 0

	def get_train_op(optimizer, loss, tvars, grad_name, if_grad_clip, **kargs):
		if if_grad_clip:
			if use_tpu:
				grads = optimizer_fn.grad_clip_fn(loss, tvars, **kargs)
				grads_and_vars = zip(grads, tvars)
			else:
				grads_and_vars = optimizer_fn.grad_clip_fn(optimizer, loss, tvars, grad_name=grad_name, **kargs)
		else:
			if use_tpu:
				grads = tf.gradients(loss, tvars)
				grads_and_vars = zip(grads, tvars)
			else:
				grads_and_vars = optimizer.compute_gradients(loss, tvars)
				grads = [grad for grad, var in grads_and_vars]
				use_norm = tf.global_norm(grads)
				tf.summary.scalar(grad_name+'/total_grad_norm', use_norm)
				for grad, var in grads_and_vars:
					if grad is not None:
						var_grad_norm = tf.global_norm([grad])
						tf.summary.scalar(grad_name+"/"+var.name, var_grad_norm)

		with tf.variable_scope(grad_name+"/"+"optimizer", reuse=tf.AUTO_REUSE):
			op = optimizer.apply_gradients(
								grads_and_vars)
		return op

	alternate_order = kargs.get("alternate_order", list(loop_step_dict.keys()))
	ebm_logz_update_circle = kargs.get("ebm_logz_update_circle", True)
	ebm_logz_update = kargs.get("ebm_logz_update", 1)

	model_cls.get_opt(optimizer_fn, init_lr_dict, optimizer_type_dict, 
						alternate_order=alternate_order,
						ebm_logz_update_circle=ebm_logz_update_circle,
						ebm_logz_update=ebm_logz_update,
						use_tpu=kargs.get('use_tpu', False))

	step2order = OrderedDict({})
	cumsum_steps = [0]+np.cumsum([loop_step_dict[key] for key in alternate_order], axis=0).tolist()

	for step, order in enumerate(alternate_order):
		step_range = list(range(cumsum_steps[step], cumsum_steps[step+1]))
		for i in step_range:
			step2order[i] = order

	print("==step2order==", step2order)

	train_op = kargs.get('train_op_type', 'joint')

	if train_op == 'alternate':
		tf.logging.info("****** alternate optimization *******")
		prev_op = tf.no_op()
		for step in range(cumsum_steps[-1]):
			order = alternate_order[step]
			with tf.control_dependencies([prev_op]):
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(update_ops):
					model_cls.get_loss(features, labels, mode, params, **kargs)
					order = step2order[step]
					if order == 'ebm':
						loss = model_cls.ebm_opt_dict['loss']
						tvars = model_cls.ebm_opt_dict['tvars']+model_cls.ebm_opt_dict['logz_tvars']
						opt = model_cls.optimizer_dict['ebm']
					elif order == 'generator':
						loss = model_cls.ebm_opt_dict['mlm_adv_loss']
						tvars = model_cls.ebm_opt_dict['mlm_tvars']
						opt = model_cls.optimizer_dict['generator']
					prev_op = get_train_op(opt, loss, tvars, order, if_grad_clip_dict[order])

	elif train_op == 'joint':
		tf.logging.info("****** joint optimization *******")
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			model_cls.get_loss(features, labels, mode, params, **kargs)
		loss = model_cls.ebm_opt_dict['loss']
		tvars = model_cls.ebm_opt_dict['tvars']+model_cls.ebm_opt_dict['logz_tvars']
		print(model_cls.ebm_opt_dict['logz_tvars'], '====logz_tvars=====')
		opt = model_cls.optimizer_dict['ebm']
		order = 'ebm'
		ebm_op = get_train_op(opt, loss, tvars, order, if_grad_clip_dict[order])
		prev_op = ebm_op

	with tf.control_dependencies([prev_op]):
		train_op = optimizer_fn.global_step.assign_add(1)

	return train_op

def token_seq_truncted(token_seq, finished_index, max_length): 
	seq_shape = bert_utils.get_shape_list(token_seq, expected_rank=[2,3])
	batch_size = seq_shape[0]
	token_seq = token_seq[:, :max_length]

	token_seq = tf.concat([token_seq, finished_index*tf.cast(tf.ones((batch_size, 1)), tf.int32)], axis=-1)

	token_seq = tf.cast(token_seq, tf.int32)
	seq_shape = bert_utils.get_shape_list(token_seq, expected_rank=[2,3])
	match_indices = tf.where(                          # [[5, 5, 2, 5, 4],
	tf.equal(finished_index, token_seq),                              #  [0, 5, 2, 3, 5],
		x=tf.range(seq_shape[1]) * tf.ones_like(token_seq),  #  [5, 1, 5, 5, 5]]
		y=(seq_shape[1])*tf.ones_like(token_seq))

	finished_pos = tf.reduce_min(match_indices, axis=1)				
	sequence_mask = tf.sequence_mask(finished_pos+1, maxlen=seq_shape[1])

	token_seq = tf.cast(sequence_mask, tf.float32) * tf.cast(token_seq, tf.float32)
				
	return tf.cast(token_seq, tf.int32)

def mixed_sample(features, mix_ratio=0.2):
	shape = bert_utils.get_shape_list(features['input_mask'], expected_rank=[2,3])
	sample_probs = tf.ones((shape[0]))
	sample_probs = mix_ratio * tf.cast(sample_probs, tf.float32) #+ 0.8 * tf.cast(must_have_one, tf.float32) # mask 15% token

	noise_dist = tf.distributions.Bernoulli(probs=sample_probs, dtype=tf.float32)
	mixed_mask = noise_dist.sample()
	mixed_mask = tf.cast(mixed_mask, tf.float32)
	return mixed_mask

def get_finised_pos_v1(token_seq, finished_index, max_length): 
	seq_shape = bert_utils.get_shape_list(token_seq, expected_rank=[2,3])
	match_indices = tf.where(                          # [[5, 5, 2, 5, 4],
	tf.equal(finished_index, token_seq),                              #  [0, 5, 2, 3, 5],
		x=tf.range(seq_shape[1]) * tf.ones_like(token_seq),  #  [5, 1, 5, 5, 5]]
		y=(seq_shape[1])*tf.ones_like(token_seq))

	finished_pos = tf.reduce_min(match_indices, axis=1)
	# sequence_mask = tf.sequence_mask(finished_pos, maxlen=max_length)
	sequence_mask = tf.cast(tf.one_hot(finished_pos, max_length), tf.float32) # [batch, max_length]
	return sequence_mask

def transfer2lm(input_ids, input_mask, finished_index=102, sentence_end_index=105):
	shape = bert_utils.get_shape_list(input_ids, expected_rank=[2,3])
	sequence_mask = get_finised_pos_v1(input_ids, finished_index, shape[1])

	sequence_mask = tf.cast(sequence_mask, tf.float32)

	modified_token_seq = tf.cast(input_ids, tf.float32) - float(finished_index) * sequence_mask + sequence_mask * sentence_end_index
	return tf.cast(modified_token_seq, tf.int32)

class EBM_NOISE_NCE(object):
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

		self.train_op_type = kargs.get('train_op_type', 'joint')
		self.ebm_prob_ln = False
		self.stop_gradient_mlm = True

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
							prob_ln=self.ebm_prob_ln,
							transform=False,
							transformer_activation="linear",
							logz_mode='none',
							normalized_constant="logv_constant_ln",
							energy_pooling="cls",
							softplus_features=False,
							use_token_type=self.model_config_dict['ebm_dist'].get('use_token_type', True),
							**kargs)

		tf.logging.info("****** using bert mlm for noise dist sample *******")

		global_step = tf.train.get_or_create_global_step()
		# self.noise_sample_ratio = tf.train.polynomial_decay(
		# 										0.25,
		# 										global_step,
		# 										self.opt_config.num_train_steps,
		# 										end_learning_rate=0.15,
		# 										power=1.0,
		# 										cycle=False)

		self.noise_sample_ratio = 0.25
	
		self.mlm_noise_dist_fn = mlm_noise_dist(self.model_config_dict['generator'],
					self.num_labels_dict['generator'],
					self.init_checkpoint_dict['generator'],
					model_reuse=None,
					load_pretrained=self.load_pretrained_dict['generator'],
					model_io_config=self.model_io_config,
					opt_config=self.opt_config,
					exclude_scope=self.exclude_scope_dict.get('generator', ""),
					not_storage_params=self.not_storage_params_dict.get('generator', []),
					target=self.target_dict['generator'],
					mask_probability=self.noise_sample_ratio,
					replace_probability=0.0,
					original_probability=0.0,
					use_token_type=self.model_config_dict['generator'].get('use_token_type', True),
					stop_gradient_mlm=self.stop_gradient_mlm,
					**kargs)

	def get_opt(self, optimizer_fn, init_lr_dict, optimizer_type_dict, **kargs):

		self.init_lr_dict = init_lr_dict
		self.optimizer_type_dict = optimizer_type_dict
		self.optimizer_dict = {}

		self.alternate_order = kargs.get('alternate_order', list(self.init_lr_dict.keys()))
		print("==alternate order==", self.alternate_order)

		for key in self.alternate_order:
			init_lr = self.init_lr_dict[key]
			optimizer_type = self.optimizer_type_dict[key]
			if optimizer_type != 'radam' and key not in ['ebm_logz']:
				learning_rate = optimizer_fn.lr_decay_fn(init_lr, self.opt_config.num_train_steps, **kargs)
				learning_rate = optimizer_fn.warm_up(learning_rate, init_lr, **kargs)
				tf.logging.info("****** leanring rate warm up:%s ******", key)
			elif key == 'ebm_logz':
				tf.logging.info("****** ebm logz learning rate ******")
				if kargs.get('ebm_logz_update_circle', False):
					lr_ratio = tf.floormod(
										tf.train.get_or_create_global_step(),
										kargs.get('ebm_logz_update', 5),
										name="ebm_logz_update"
									)
					lr_ratio = tf.cast(tf.equal(tf.cast(lr_ratio, tf.int32), 0), tf.float32)
					tf.logging.info("****** learning_rate circle update ****** with %s circle", kargs.get('ebm_logz_update', 5))
				else:
					lr_ratio = 1.0
					tf.logging.info("****** normal learning_rate ******")
				if not kargs.get("use_tpu", False):
					tf.summary.scalar('{}_lr_ratio'.format(key), lr_ratio)
				learning_rate = init_lr * lr_ratio
			if not kargs.get("use_tpu", False):
				tf.summary.scalar('{}_learning_rate'.format(key), learning_rate)

			tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
			opt = optimizer_fn.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

			if kargs.get("use_tpu", False):
				tf.logging.info("***** Using tpu cross shard optimizer *****")
				opt = tf.contrib.tpu.CrossShardOptimizer(opt)
			self.optimizer_dict[key] = opt

	def get_loss(self, features, labels, mode, params, **kargs):
		true_features = {}

		for key in features:
			if key == 'input_ori_ids':
				true_features["input_ids"] = tf.cast(features['input_ori_ids'], tf.int32)
				# true_features["input_ids"] = transfer2lm(true_features['input_ids'], 
				# 										features['input_mask'])
			if key in ['input_mask', 'segment_ids']:
				true_features[key] = tf.cast(features[key], tf.int32)

		self.mlm_noise_dist_dict = self.mlm_noise_dist_fn(features, labels, mode, params)

		# third, get fake ebm dict
		fake_features = {}
		fake_features["input_ids"] = self.mlm_noise_dist_dict['sampled_ids']
		fake_features["input_mask"] = self.mlm_noise_dist_dict['sampled_mask']
		# fake_features['input_mask'] = tf.cast(features['input_mask'], tf.int32)
		fake_features['segment_ids'] = tf.zeros_like(features['input_mask'])
		tf.logging.info("****** using bert mlm stop gradient *******")

		# second, get true ebm dict
		self.true_ebm_dist_dict = self.ebm_dist_fn(true_features, labels, mode, params)
		self.fake_ebm_dist_dict = self.ebm_dist_fn(fake_features, labels, mode, params)

		# self.ebm_loss = get_residual_ebm_loss(self.true_ebm_dist_dict['logits'], 
		# 						self.fake_ebm_dist_dict['logits'], 
		# 						use_tpu=kargs.get('use_tpu', False))

		self.gan_type = kargs.get('gan_type', 'JS')
		tf.logging.info("****** gan type *******", self.gan_type)

		[self.ebm_loss, self.mlm_adv_loss] = get_ebm_mlm_adv_loss(self.true_ebm_dist_dict['logits'], 
															self.fake_ebm_dist_dict['logits'], 
															gan_type=self.gan_type,
															use_tpu=kargs.get('use_tpu', False))
		
		self.true_ebm_dist_dict['logz_loss'] = self.ebm_loss

		self.ebm_opt_dict = {
			"loss":self.ebm_loss,
			"tvars":self.true_ebm_dist_dict['tvars'],
			"logz_tvars":self.true_ebm_dist_dict['logz_tvars'],
			"logz_loss":self.true_ebm_dist_dict['logz_loss'],
			"mlm_adv_loss":self.mlm_adv_loss,
			"mlm_tvars":self.mlm_noise_dist_dict['tvars']
		}

		self.loss = self.ebm_loss
		self.tvars = self.true_ebm_dist_dict['logz_tvars'] + self.true_ebm_dist_dict['tvars'] + self.mlm_noise_dist_dict['tvars']

	def load_pretrained_model(self, **kargs):
		self.var_checkpoint_dict_list = []
		for key in self.init_checkpoint_dict:
			if self.load_pretrained_dict[key] == "yes":
				if key == 'ebm_dist':
					tmp = {
							"tvars":self.ebm_opt_dict['tvars']+self.ebm_opt_dict['logz_tvars'],
							"init_checkpoint":self.init_checkpoint_dict['ebm_dist'],
							"exclude_scope":self.exclude_scope_dict[key],
							"restore_var_name":self.model_config_dict['ebm_dist'].get('restore_var_name', [])
					}
					if kargs.get("sharing_mode", "none") != "none":
						tmp['exclude_scope'] = ''
					self.var_checkpoint_dict_list.append(tmp)
				elif key == 'generator':
					tmp = {
							"tvars":self.mlm_noise_dist_dict['tvars'],
							"init_checkpoint":self.init_checkpoint_dict['generator'],
							"exclude_scope":self.exclude_scope_dict[key],
							"restore_var_name":self.model_config_dict['generator'].get('restore_var_name', [])
					}
					if kargs.get("sharing_mode", "none") != "none":
						tmp['exclude_scope'] = ''
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

		train_op_type = kargs.get('train_op_type', 'joint')

		ebm_noise_fce = EBM_NOISE_NCE(model_config_dict,
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
		use_tpu = 1 if kargs.get('use_tpu', False) else 0
			
		if mode == tf.estimator.ModeKeys.TRAIN:

			if kargs.get('use_tpu', False):
				optimizer_fn = optimizer.Optimizer(opt_config)
				use_tpu = 1
			else:
				optimizer_fn = distributed_optimizer.Optimizer(opt_config)
				use_tpu = 0

			train_op = get_train_op(
								ebm_noise_fce, 
								optimizer_fn, 
								opt_config,
								model_config_dict['ebm_dist'], 
								model_config_dict['noise_dist'],
								model_config_dict['generator'],
								features, labels, mode, params,
								use_tpu=use_tpu,
								train_op_type=train_op_type,
								alternate_order=['ebm'])

			ebm_noise_fce.load_pretrained_model(**kargs)
			var_checkpoint_dict_list = ebm_noise_fce.var_checkpoint_dict_list
			loss = ebm_noise_fce.loss
			tvars = ebm_noise_fce.tvars

			if len(var_checkpoint_dict_list) >= 1:
				scaffold_fn = model_io_fn.load_multi_pretrained(
												var_checkpoint_dict_list,
												use_tpu=use_tpu)
			else:
				scaffold_fn = None

			metric_dict = ebm_train_metric(
										ebm_noise_fce.true_ebm_dist_dict['logits'], 
										ebm_noise_fce.fake_ebm_dist_dict['logits']
										)

			if not kargs.get('use_tpu', False):
				for key in metric_dict:
					tf.summary.scalar(key, metric_dict[key])
				tf.summary.scalar("ebm_loss", ebm_noise_fce.ebm_opt_dict['loss'])
	
			model_io_fn.print_params(tvars, string=", trainable params")

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
			ebm_noise_fce.load_pretrained_model(**kargs)
			var_checkpoint_dict_list = ebm_noise_fce.var_checkpoint_dict_list
			loss = ebm_noise_fce.loss

			if len(var_checkpoint_dict_list) >= 1:
				scaffold_fn = model_io_fn.load_multi_pretrained(
												var_checkpoint_dict_list,
												use_tpu=use_tpu)
			else:
				scaffold_fn = None

			tpu_eval_metrics = (ebm_eval_metric, 
								[
								ebm_noise_fce.true_ebm_dist_dict['logits'], 
								ebm_noise_fce.fake_ebm_dist_dict['logits']
								])
			gpu_eval_metrics = ebm_eval_metric(
								ebm_noise_fce.true_ebm_dist_dict['logits'], 
								ebm_noise_fce.fake_ebm_dist_dict['logits']
								)

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


