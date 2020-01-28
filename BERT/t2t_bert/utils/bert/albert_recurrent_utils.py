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

from utils.bert import bert_utils
from utils.bert import layer_norm_utils
from utils.bert import bert_adapter_modules
from utils.bert import albert_modules

def transformer_cell(input_tensor,
						attention_mask=None,
						hidden_size=768,
						num_hidden_layers=12,
						num_attention_heads=12,
						intermediate_size=3072,
						intermediate_act_fn=gelu,
						hidden_dropout_prob=0.1,
						attention_probs_dropout_prob=0.1,
						initializer_range=0.02,
						do_return_all_layers=False,
						shared_type=None,
						adapter_fn=None):

	layer_input = bert_utils.reshape_to_matrix(input_tensor)
	with tf.variable_scope("layer_shared", reuse=tf.AUTO_REUSE):
		with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
			attention_heads = []
			with tf.variable_scope("self"):
				[attention_head, 
				attention_scores] = albert_modules.attention_layer(
						from_tensor=layer_input,
						to_tensor=layer_input,
						attention_mask=attention_mask,
						num_attention_heads=num_attention_heads,
						size_per_head=attention_head_size,
						attention_probs_dropout_prob=attention_probs_dropout_prob,
						initializer_range=initializer_range,
						do_return_2d_tensor=True,
						batch_size=batch_size,
						from_seq_length=seq_length,
						to_seq_length=seq_length)
				attention_heads.append(attention_head)
				all_attention_scores.append(attention_scores)

			attention_output = None
			if len(attention_heads) == 1:
				attention_output = attention_heads[0]
			else:
				# In the case where we have other sequences, we just concatenate
				# them to the self-attention head before the projection.
				attention_output = tf.concat(attention_heads, axis=-1)

			# Run a linear projection of `hidden_size` then add a residual
			# with `layer_input`.
			with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
				attention_output = tf.layers.dense(
						attention_output,
						hidden_size,
						kernel_initializer=albert_modules.create_initializer(initializer_range))
				attention_output = albert_modules.dropout(attention_output, hidden_dropout_prob)

				if adapter_fn:
					attention_output = adapter_fn(attention_output, 
												layer_idx=layer_idx)

				attention_output = albert_modules.layer_norm(attention_output + layer_input)

		# The activation is only applied to the "intermediate" hidden layer.
		with tf.variable_scope('intermediate', reuse=tf.AUTO_REUSE):
			intermediate_output = tf.layers.dense(
					attention_output,
					intermediate_size,
					activation=intermediate_act_fn,
					kernel_initializer=albert_modules.create_initializer(initializer_range))

		# Down-project back to `hidden_size` then add the residual.
		with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
			layer_output = tf.layers.dense(
					intermediate_output,
					hidden_size,
					kernel_initializer=albert_modules.create_initializer(initializer_range))
			layer_output = albert_modules.dropout(layer_output, hidden_dropout_prob)

		if adapter_fn:
			layer_output = adapter_fn(attention_output, 
									layer_idx=layer_idx)

		layer_output = albert_modules.layer_norm(layer_output + attention_output)
