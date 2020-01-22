
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils.bert import albert_utils_official
from utils.bert import albert_modules_official
import copy
import collections
import json
import math
import re
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

class Albert(object):
	def __init__(self, config, *args, **kargs):
		self.config = copy.deepcopy(config)
		tf.logging.info(" begin to build {}".format(self.config.get("scope", "bert")))

	def build_embedder(self, input_ids, token_type_ids, 
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									**kargs):

		reuse = kargs["reuse"]
		input_shape = albert_utils_official.get_shape_list(input_ids, expected_rank=[2,3])
		batch_size = input_shape[0]
		seq_length = input_shape[1]

		if token_type_ids is None:
			token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

		if self.config.get('embedding_scope', None):
			embedding_scope = self.config['embedding_scope']
			tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s==", embedding_scope)
		else:
			embedding_scope = self.config.get("scope", "bert")
			tf.logging.info("==using embedding scope of original model_config.scope: %s==", embedding_scope)

		with tf.variable_scope(embedding_scope, reuse=reuse):
			with tf.variable_scope("embeddings"):
				# Perform embedding lookup on the word ids.

				input_shape = albert_utils_official.get_shape_list(input_ids, expected_rank=[2,3])
				if len(input_shape) == 3:
					tf.logging.info("****** 3D embedding matmul *******")
					(self.embedding_output_word, self.embedding_table) = albert_modules_official.gumbel_embedding_lookup(
							input_ids=input_ids,
							vocab_size=self.config.vocab_size,
							embedding_size=self.config.embedding_size,
							initializer_range=self.config.initializer_range,
							word_embedding_name="word_embeddings",
							use_one_hot_embeddings=self.config.use_one_hot_embeddings)
				elif len(input_shape) == 2:
					(self.embedding_output_word, self.embedding_table) = albert_modules_official.embedding_lookup(
						input_ids=input_ids,
						vocab_size=self.config.vocab_size,
						embedding_size=self.config.embedding_size,
						initializer_range=self.config.initializer_range,
						word_embedding_name="word_embeddings",
						use_one_hot_embeddings=self.config.use_one_hot_embeddings)
				else:
					(self.embedding_output_word, self.embedding_table) = albert_modules_official.embedding_lookup(
						input_ids=input_ids,
						vocab_size=self.config.vocab_size,
						embedding_size=self.config.embedding_size,
						initializer_range=self.config.initializer_range,
						word_embedding_name="word_embeddings",
						use_one_hot_embeddings=self.config.use_one_hot_embeddings)

				if kargs.get("perturbation", None):
					self.embedding_output_word += kargs["perturbation"]
					tf.logging.info(" add word pertubation for robust learning ")

				# Add positional embeddings and token type embeddings, then layer
				# normalize and perform dropout.
				self.embedding_output = albert_modules_official.embedding_postprocessor(
						input_tensor=self.embedding_output_word,
						use_token_type=True,
						token_type_ids=token_type_ids,
						token_type_vocab_size=self.config.type_vocab_size,
						token_type_embedding_name="token_type_embeddings",
						use_position_embeddings=True,
						position_embedding_name="position_embeddings",
						initializer_range=self.config.initializer_range,
						max_position_embeddings=self.config.max_position_embeddings,
						dropout_prob=hidden_dropout_prob)

	def build_encoder(self, input_ids, input_mask, 
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									**kargs):
		reuse = kargs["reuse"]
		input_shape = albert_utils_official.get_shape_list(input_ids, expected_rank=[2,3])
		batch_size = input_shape[0]
		seq_length = input_shape[1]
		if input_mask is None:
			input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
		with tf.variable_scope(self.config.get("scope", "bert"), reuse=reuse):
			with tf.variable_scope("encoder"):
				# This converts a 2D mask of shape [batch_size, seq_length] to a 3D
				# mask of shape [batch_size, seq_length, seq_length] which is used
				# for the attention scores.
				
				# Run the stacked transformer.
				# `sequence_output` shape = [batch_size, seq_length, hidden_size].

				if kargs.get('attention_type', 'efficient_attention') == 'normal_attention':
					tf.logging.info("****** normal attention *******")
					transformer_model = albert_modules_official.transformer_model
				else:
					tf.logging.info("****** normal attention *******")
					transformer_model = albert_modules_official.transformer_model

				[self.all_encoder_layers,
				self.all_attention_scores] = albert_modules_official.transformer_model(
						input_tensor=self.embedding_output,
						attention_mask=input_mask,
						hidden_size=self.config.hidden_size,
						num_hidden_layers=self.config.num_hidden_layers,
						num_hidden_groups=self.config.num_hidden_groups,
						num_attention_heads=self.config.num_attention_heads,
						intermediate_size=self.config.intermediate_size,
						inner_group_num=self.config.inner_group_num,
						intermediate_act_fn=albert_modules_official.get_activation(self.config.hidden_act),
						hidden_dropout_prob=self.config.hidden_dropout_prob,
						attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
						initializer_range=self.config.initializer_range,
						do_return_all_layers=True)

	def build_pooler(self, *args,**kargs):
		reuse = kargs["reuse"]
		layer_num = kargs.get("layer_num", -1)
		with tf.variable_scope(self.config.get("scope", "bert"), reuse=reuse):
			# self.sequence_output = self.all_encoder_layers[-1]
			self.sequence_output = self.get_encoder_layers(layer_num)

			# The "pooler" converts the encoded sequence tensor of shape
			# [batch_size, seq_length, hidden_size] to a tensor of shape
			# [batch_size, hidden_size]. This is necessary for segment-level
			# (or segment-pair-level) classification tasks where we need a fixed
			# dimensional representation of the segment.
			with tf.variable_scope("pooler"):
				# We "pool" the model by simply taking the hidden state corresponding
				# to the first token. We assume that this has been pre-trained
				first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
				self.pooled_output = tf.layers.dense(
						first_token_tensor,
						self.config.hidden_size,
						activation=tf.tanh,
						kernel_initializer=albert_modules_official.create_initializer(self.config.initializer_range))

	def get_multihead_attention(self):
		return self.all_attention_scores
	
	def get_pooled_output(self):
		return self.pooled_output

	def get_embedding_projection_table(self):
		return None

	def get_sequence_output(self):
		"""Gets final hidden layer of encoder.

		Returns:
			float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
			to the final hidden of the transformer encoder.
		"""
		return self.sequence_output

	def get_all_encoder_layers(self):
		return self.all_encoder_layers

	def get_embedding_table(self):
		return self.embedding_table

	def get_encoder_layers(self, layer_num):
		if layer_num >= 0 and layer_num <= len(self.all_encoder_layers) - 1:
			print("==get encoder layer==", layer_num)
			return self.all_encoder_layers[layer_num]
		else:
			return self.all_encoder_layers[-1]