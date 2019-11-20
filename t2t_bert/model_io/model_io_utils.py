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
import numpy as np

class RestoreParametersAverageValues(tf.train.SessionRunHook):
	"""
	Replace parameters with their moving averages.
	This operation should be executed only once, and before any inference.
	"""
	def __init__(self, ema):
		"""
		:param ema:         tf.train.ExponentialMovingAverage
		"""
		super(RestoreParametersAverageValues, self).__init__()
		self._ema = ema
		self._restore_ops = None

	def begin(self):
		""" Create restoring operations before the graph been finalized. """
		ema_variables = tf.moving_average_variables()
		self._restore_ops = [tf.assign(x, self._ema.average(x)) for x in ema_variables]
		print("==get restore ops==")

	def after_create_session(self, session, coord):
		""" Restore the parameters right after the session been created. """
		print("==restore ema variables==")
		session.run(self._restore_ops)

def ema_saver():	
	ema = tf.train.ExponentialMovingAverage(0.99)
	saver = tf.train.Saver(ema.variables_to_restore())
	return saver
	
def ema_getter(getter, name, *args, **kwargs):
	'''
	http://ruishu.io/2017/11/22/ema/
	'''
	var = getter(name, *args, **kwargs)
	ema_var = ema.average(var)
	return ema_var if ema_var else var

def track_params_averages(params_moving_average_decay, scope, **kargs):
	'''
	https://github.com/tensorflow/tensorflow/issues/3460
	'''
	"""
	Track the moving averages of parameters.
	Must be invoked after `infer()` and before `train()`.

	:return:
			ema:                    `tf.train.ExponentialMovingAverage`
			params_averages_op:     operator that tracking averages
	add two stage ema
	"""
	global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
	
	if kargs.get("two_stage", False):
		cond_fn = tf.less(global_step, tf.constant(kargs.get('first_stage_steps', -1), dtype=tf.float32))
		decay_beta = tf.minimum(params_moving_average_decay, (global_step-kargs.get('first_stage_steps', -1))/(global_step-kargs.get('first_stage_steps', -1)+1))
		decay_beta_final = tf.cond(cond_fn,
									lambda:tf.constant(value=0.0, shape=[], dtype=tf.float32, name="first_stage_decay"),
									lambda:decay_beta)
	else:
		decay_beta_final = tf.minimum(params_moving_average_decay, (global_step)/(global_step+1))

	tf.summary.scalar('ema_decay_rate', decay_beta_final)

	ema = tf.train.ExponentialMovingAverage(decay=decay_beta_final)
	return ema

def print_params(tvars, string):
	for var in tvars:
		tf.logging.info(" name = %s, shape = %s%s", 
						var.name, var.shape, string)

def count_variables(scope, **kargs):
	not_storage_params = kargs.get("not_storage_params", [])
	tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
	total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tvars])
	return total_params
	
def get_params(scope, **kargs):
	not_storage_params = kargs.get("not_storage_params", [])
	collections = kargs.get("collections", [])
	if len(collections) >= 1:
		tvars = []
		for collection in collections:
			tvars += tf.get_collection(collection)
	else:
		tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
	if len(not_storage_params) >= 1:
		storage_tvars = []
		for var in tvars:
			var_name_lst = var.name.split("/")
			interaction = set(var_name_lst) & set(not_storage_params)
			if len(interaction) >= 1:
				continue
			else:
				storage_tvars.append(var)
	else:
		storage_tvars = tvars
	return storage_tvars

def init_pretrained(assignment_map, initialized_variable_names,
										tvars, init_checkpoint, **kargs):
	tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
	for var in tvars:
		init_string = ""
		init_checkpoint_string = ""
		if var.name in initialized_variable_names:
			init_string = ", *INIT_FROM_CKPT*"
			init_checkpoint_string = init_checkpoint
		
		tf.logging.info(" name = %s, shape = %s%s, from checkpoint = %s", 
						var.name, var.shape, init_string, init_checkpoint_string)

def get_actual_scope(name, exclude_scope):
	return "/".join([exclude_scope, name])

def get_assigment_map_from_checkpoint(tvars, init_checkpoint, **kargs):
	"""Compute the union of the current variables and checkpoint variables."""
	assignment_map = {}
	initialized_variable_names = {}

	exclude_scope = kargs.get("exclude_scope", "")

	name_to_variable = collections.OrderedDict()
	for var in tvars:
		name = var.name
		m = re.match("^(.*):\\d+$", name)
		if m is not None:
			name = m.group(1)
		name_to_variable[name] = var

	init_vars = tf.train.list_variables(init_checkpoint)

	assignment_map = collections.OrderedDict()
	for x in init_vars:
		(name, var) = (x[0], x[1])
		if len(exclude_scope) >= 1:
			assignment_name = get_actual_scope(name, exclude_scope)
		else:
			assignment_name = name

		if assignment_name not in name_to_variable:
			continue
		assignment_map[name] = assignment_name
		initialized_variable_names[assignment_name] = 1
		initialized_variable_names[assignment_name + ":0"] = 1

	return (assignment_map, initialized_variable_names)

def apply_ema(tvars, loss, decay=0.999, **kargs):

	var_ema = tf.train.ExponentialMovingAverage(decay)
	ema_op = var_ema.apply(tvars)
	with tf.control_dependencies([ema_op]):
		loss = tf.identity(loss)

		shadow_vars = []
		global_vars = []
		for var in tvars:
			v = var_ema.average(var)
			if v:
				shadow_vars.append(v)
				global_vars.append(var)
		assign_vars = []
		for g,v in zip(global_vars, shadow_vars):
			assign_vars.append(tf.assign(g,v))
	return assign_vars





	