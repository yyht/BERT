
import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np

from utils.bert import bert_utils

class DropoutContext(object):
	def __init__(self):
		self.dropout = 0
		self.mask = None
		self.scale = 1
		self.reuse_mask = True
		self.noise_shape = None
		self.seed = None

class XDropout(object):
	def get_mask(self, input_tensor, local_context):

		if not isinstance(local_context, DropoutContext):
			dropout = local_context
			mask = None
			noise_shape = None
			seed = None
			tf.logging.info("==not reuse dropout mask==")
		else:
			dropout = local_context.dropout
			dropout *= local_context.scale
			mask = local_context.mask if local_context.reuse_mask else None
			noise_shape = local_context.noise_shape
			seed = local_context.seed
			tf.logging.info("==reuse dropout mask==")

		if dropout > 0 and mask is None:
			if not noise_shape:
				noise_shape = bert_utils.get_shape_list(input_tensor)
			random_tensor = tf.random_uniform(
					noise_shape, seed=seed, 
					dtype=input_tensor.dtype)
			mask = tf.cast(random_tensor > dropout, dtype=tf.float32)
			tf.logging.info("==generate new mask==")

		if isinstance(local_context, DropoutContext):
			if local_context.mask is None:
				local_context.mask = mask
				tf.logging.info("==push mask==")
			if local_context.noise_shape is None:
				local_context.noise_shape = noise_shape
				tf.logging.info("==push noise shape==")

		return mask, dropout

	def dropout(self, input_tensor, local_context):
		mask, dropout = self.get_mask(input_tensor, local_context)
		scale = 1.0 / (1.0-dropout)
		if dropout > 0:
			output = input_tensor * scale * mask
		else:
			output = input_tensor
		return output

class ReuseDropout(object):
	def __init__(self):
		self.context_stack = {}

	def get_context(self, dropout_prob,
						context_name=None,
						noise_shape=None,
						seed=None):
		if context_name:
			if context_name not in self.context_stack:
				self.context_stack[context_name] = DropoutContext()
				tf.logging.info("==add new dropout context: %s==" % (context_name))
			ctx = self.context_stack[context_name]
			ctx.dropout = dropout_prob
			ctx.noise_shape = noise_shape
			ctx.seed = seed
			return ctx
		else:
			return dropout_prob

	def dropout(self, input_tensor, dropout_prob, 
						context_name=None,
						noise_shape=None,
						seed=None):
		if dropout_prob > 0:
			dropout_fn = XDropout()
			output = dropout_fn.dropout(input_tensor, 
					self.get_context(dropout_prob,
									context_name,
									noise_shape,
									seed))
			return output
		else:
			return input_tensor
						

