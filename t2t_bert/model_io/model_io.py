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
from model_io import model_io_utils

class ModelIO(object):
	def __init__(self, config, **kargs):
		print(" initializing ModelIO ")
		self.config = config

	def get_ema_hooks(self, train_op, var_list, params_moving_average_decay, scope, mode,
				**kargs):
		self.ema = model_io_utils.track_params_averages(
								params_moving_average_decay, 
								scope,
								**kargs)
		if mode == tf.estimator.ModeKeys.TRAIN:
			with tf.control_dependencies([train_op]):
				if not var_list:
					tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				else:
					tvars = var_list
				params_averages_op = self.ema.apply(tvars)
			return params_averages_op, None
			# tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.group(params_averages_op))
		elif mode == tf.estimator.ModeKeys.EVAL or tf.estimator.ModeKeys.PREDICT:
			hooks = model_io_utils.RestoreParametersAverageValues(self.ema)
			return None, hooks
		else:
			return None, None

	def get_ema_saver(self):
		self.ema_saver = model_io_utils.ema_saver()

	def moving_average_saver(self, opt, **kargs):
		self.saver = opt.swapping_saver(var_list=kargs.get("var_lst", None),
								max_to_keep=self.config.get("max_to_keep", 100))
		
	def set_saver(self, opt=None, **kargs):
		if len(kargs.get("var_lst", [])) >= 1:
			self.saver = tf.train.Saver(var_list=kargs.get("var_lst", None),
			max_to_keep=self.config.get("max_to_keep", 100))
		else:
			self.saver = tf.train.Saver(
			max_to_keep=self.config.get("max_to_keep", 100))

		if self.config.get("ema_saver", "no") == "yes":
			# try:
			print("==apply ema saver==")
			self.moving_average_saver(opt, **kargs)
			# except:
				# print("==no valid eam saver==")

	def get_hooks(self, checkpoint_dir, num_storage_steps):
		self.checkpoint_hook = [tf.train.CheckpointSaverHook(
									checkpoint_dir,
									save_secs=None,
									save_steps=num_storage_steps,
									saver=self.saver,
									checkpoint_basename='model.ckpt',
									scaffold=None,
									listeners=None
								)]

	def init_model(self, sess, **kargs):
		sess.run(tf.global_variables_initializer())

	def save_model(self, sess, checkpoint, **kargs):
		self.saver.save(sess, checkpoint)

	def load_model(self, sess, checkpoint, **kargs):
		self.saver.restore(sess, checkpoint)

	def apply_ema(self, sess, tvars, loss, **kargs):
		decay = self.config.get("ema_decay", 0.999)
		assign_vars = model_io_utils.apply_ema(tvars, loss, 
											decay=decay, **kargs)
		
	def get_params(self, scope, **kargs):
		tvars = model_io_utils.get_params(scope, **kargs)
		return tvars

	def count_params(self, scope, **kargs):
		return model_io_utils.count_variables(scope, **kargs)

	def print_params(self, tvars, string):
		model_io_utils.print_params(tvars, string)

	def load_pretrained(self, tvars, init_checkpoint, **kargs):
		print(kargs.get("exclude_scope", ""), "===============")
		[assignment_map, 
		initialized_variable_names] = model_io_utils.get_assigment_map_from_checkpoint(
															tvars, 
															init_checkpoint,
															**kargs)

		scaffold_fn = None
		if kargs.get('use_tpu', 0) == 0:

			model_io_utils.init_pretrained(assignment_map, 
											initialized_variable_names,
											tvars, init_checkpoint, **kargs)
			print("==succeeded in loading pretrained model==")
		else:
			tf.logging.info(" initializing parameter from init checkpoint ")
			def tpu_scaffold():
				model_io_utils.init_pretrained(assignment_map, 
											initialized_variable_names,
											tvars, init_checkpoint, **kargs)
				# tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
				return tf.train.Scaffold()
			scaffold_fn = tpu_scaffold 

		return scaffold_fn

	def load_multi_pretrained(self, var_checkpoint_dict_list, **kargs):
		print(kargs.get("exclude_scope", ""), "===============")
		def init_multi_model(var_checkpoint_dict_list):
			for item in var_checkpoint_dict_list:
				tvars = item['tvars']
				init_checkpoint = item['init_checkpoint']
				exclude_scope = item['exclude_scope']
				restore_var_name = item.get('restore_var_name', [])
				[assignment_map, 
				initialized_variable_names] = model_io_utils.get_assigment_map_from_checkpoint(
																	tvars, 
																	init_checkpoint, 
																	exclude_scope=exclude_scope,
																	restore_var_name=restore_var_name)
				model_io_utils.init_pretrained(assignment_map, 
											initialized_variable_names,
											tvars, init_checkpoint, **kargs)

		scaffold_fn = None
		if kargs.get('use_tpu', 0) == 0:
			init_multi_model(var_checkpoint_dict_list)
		else:
			tf.logging.info(" initializing parameter from init checkpoint ")
			def tpu_scaffold():
				init_multi_model(var_checkpoint_dict_list)
				return tf.train.Scaffold()
			scaffold_fn = tpu_scaffold
		return scaffold_fn







