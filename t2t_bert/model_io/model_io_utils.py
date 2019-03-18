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

def print_params(tvars, string):
	for var in tvars:
		tf.logging.info(" name = %s, shape = %s%s", 
						var.name, var.shape, string)

def count_variables(scope, **kargs):
	not_storage_params = kargs.get("not_storage_params", [])
	tvars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope)
	print(np.sum([np.prod(v.get_shape().as_list()) for v in tvars]))
	
def get_params(scope, **kargs):
	not_storage_params = kargs.get("not_storage_params", [])
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
		if var.name in initialized_variable_names:
			init_string = ", *INIT_FROM_CKPT*"
		tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape,
										init_string)

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