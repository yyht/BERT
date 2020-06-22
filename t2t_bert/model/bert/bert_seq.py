import numpy as np
import tensorflow as tf
from utils.bert import bert_utils
from utils.bert import bert_seq_modules
from utils.bert import bert_modules
import copy

class Bert(object):
	"""
	default scope: bert
	"""
	def __init__(self, config, *args, **kargs):
		self.config = copy.deepcopy(config)
		tf.logging.info(" begin to build {}".format(self.config.get("scope", "bert")))

	def build_embedder(self, input_ids, token_type_ids, 
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									past=None,
									decode_loop_step=None,
									**kargs):

		reuse = kargs["reuse"]
		embedding_table_adv = kargs.get('embedding_table_adv', None)
		print(embedding_table_adv, "==embedding-adv")

		if self.config.get("embedding", "none_factorized") == "none_factorized":
			projection_width = self.config.hidden_size
			tf.logging.info("==not using embedding factorized==")
		else:
			projection_width = self.config.get('embedding_size', self.config.hidden_size)
			tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

		if self.config.get('embedding_scope', None):
			embedding_scope = self.config['embedding_scope']
			other_embedding_scope = self.config['embedding_scope'] #self.config.get("scope", "bert")
			tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s, other_embedding_scope:%s ==", embedding_scope, other_embedding_scope)
		else:
			embedding_scope = self.config.get("scope", "bert")
			other_embedding_scope = self.config.get("scope", "bert")
			tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s, other_embedding_scope:%s ==", embedding_scope, other_embedding_scope)
		if past is None:
			self.past_length = 0
		else:
			# batch_size_, num_layers_, two_, num_heads_, self.cache_length, features_
			if decode_loop_step is None:
				# gpu-decode length
				past_shape = bert_utils.get_shape_list(past, expected_rank=[6])
				self.past_length = past_shape[-2]
			else:
				self.past_length = decode_loop_step

		with tf.variable_scope(embedding_scope, reuse=reuse):
			with tf.variable_scope("embeddings"):
				# Perform embedding lookup on the word ids.
				# (self.embedding_output_word, self.embedding_table) = bert_modules.embedding_lookup(
				# 		input_ids=input_ids,
				# 		vocab_size=self.config.vocab_size,
				# 		embedding_size=projection_width,
				# 		initializer_range=self.config.initializer_range,
				# 		word_embedding_name="word_embeddings",
				# 		use_one_hot_embeddings=self.config.use_one_hot_embeddings)

				input_shape = bert_utils.get_shape_list(input_ids, expected_rank=[2,3])
				print(input_shape, "=====input_shape=====")
				if len(input_shape) == 3:
					tf.logging.info("****** 3D embedding matmul *******")
					(self.embedding_output_word, self.embedding_table) = bert_modules.gumbel_embedding_lookup(
							input_ids=input_ids,
							vocab_size=self.config.vocab_size,
							embedding_size=projection_width,
							initializer_range=self.config.initializer_range,
							word_embedding_name="word_embeddings",
							use_one_hot_embeddings=self.config.use_one_hot_embeddings,
							embedding_table_adv=embedding_table_adv)
				elif len(input_shape) == 2:
					(self.embedding_output_word, self.embedding_table) = bert_modules.embedding_lookup(
						input_ids=input_ids,
						vocab_size=self.config.vocab_size,
						embedding_size=projection_width,
						initializer_range=self.config.initializer_range,
						word_embedding_name="word_embeddings",
						use_one_hot_embeddings=self.config.use_one_hot_embeddings,
						embedding_table_adv=embedding_table_adv)
				else:
					(self.embedding_output_word, self.embedding_table) = bert_modules.embedding_lookup(
						input_ids=input_ids,
						vocab_size=self.config.vocab_size,
						embedding_size=projection_width,
						initializer_range=self.config.initializer_range,
						word_embedding_name="word_embeddings",
						use_one_hot_embeddings=self.config.use_one_hot_embeddings,
						embedding_table_adv=embedding_table_adv)

				# if kargs.get("perturbation", None):
				# 	self.embedding_output_word += kargs["perturbation"]
				# 	tf.logging.info(" add word pertubation for robust learning ")

		with tf.variable_scope(other_embedding_scope, reuse=reuse):
			with tf.variable_scope("embeddings"):

				if kargs.get("reuse_mask", False):
					dropout_name = other_embedding_scope + "/embeddings"
					tf.logging.info("****** reuse mask: %s *******".format(dropout_name))
				else:
					dropout_name = None

				# Add positional embeddings and token type embeddings, then layer
				# normalize and perform dropout.
				tf.logging.info("==using segment type embedding ratio: %s==", str(self.config.get("token_type_ratio", 1.0)))
				self.embedding_output = bert_seq_modules.embedding_postprocessor(
						input_tensor=self.embedding_output_word,
						use_token_type=kargs.get('use_token_type', True),
						token_type_ids=token_type_ids,
						token_type_vocab_size=self.config.type_vocab_size,
						token_type_embedding_name="token_type_embeddings",
						use_position_embeddings=True,
						position_embedding_name="position_embeddings",
						initializer_range=self.config.initializer_range,
						max_position_embeddings=self.config.max_position_embeddings,
						dropout_prob=hidden_dropout_prob,
						token_type_ratio=self.config.get("token_type_ratio", 1.0),
						position_offset=self.past_length,
						dropout_name=dropout_name)

	def build_encoder(self, input_ids, input_mask, 
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									embedding_output=None,
									past=None,
									decode_loop_step=None,
									max_decode_length=None,
									if_bp=False,
									if_cache_decode=None,
									**kargs):
		reuse = kargs["reuse"]
		input_shape = bert_utils.get_shape_list(input_ids, expected_rank=[2,3])
		batch_size = input_shape[0]
		seq_length = input_shape[1]

		if input_mask is None:
			input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

		with tf.variable_scope(self.config.get("scope", "bert"), reuse=reuse):
			with tf.variable_scope("encoder"):
				# This converts a 2D mask of shape [batch_size, seq_length] to a 3D
				# mask of shape [batch_size, seq_length, seq_length] which is used
				# for the attention scores.

				input_shape = bert_utils.get_shape_list(input_ids, expected_rank=[2,3])
				if len(input_shape) == 3:
					tmp_input_ids = tf.argmax(input_ids, axis=-1)
				else:
					tmp_input_ids = input_ids

				if decode_loop_step is None:
					self.bi_attention_mask = bert_seq_modules.create_attention_mask_from_input_mask(
						tmp_input_ids, input_mask)
				else:
					if max_decode_length is None:
						max_decode_length = self.max_position_embeddings
					# [max_decode_length, 1]
					input_mask = tf.expand_dims(tf.sequence_mask(decode_loop_step+1, maxlen=max_decode_length), axis=-1)
					# [1, max_decode_length]
					input_mask = tf.transpose(input_mask, perm=[1,0])
					input_mask = tf.tile(input_mask, [batch_size, 1])
					self.bi_attention_mask = bert_seq_modules.create_attention_mask_from_input_mask(
						tmp_input_ids, input_mask)

				seq_type = kargs.get('seq_type', "None")
				print(seq_type)

				if seq_type == "seq2seq":
					if kargs.get("mask_type", "left2right") == "left2right":
						mask_sequence = None
						tf.logging.info("==apply left2right LM model with casual mask==")
					elif kargs.get("mask_type", "left2right") == "seq2seq":
						token_type_ids = kargs.get("token_type_ids", None)
						tf.logging.info("==apply left2right LM model with conditional casual mask==")
						if token_type_ids is None:
							token_type_ids = tf.zeros_like(input_mask)
							tf.logging.info("==conditional mask is set to 0 and degenerate to left2right LM model==")
						mask_sequence = token_type_ids
					else:
						mask_sequence = None
					if decode_loop_step is None:
						self.attention_mask = bert_utils.generate_seq2seq_mask(self.bi_attention_mask, 
														mask_sequence,
														seq_type)
					else:
						# with loop step, we must do casual decoding
						self.attention_mask = bert_utils.generate_seq2seq_mask(self.bi_attention_mask, 
														None,
														seq_type)
				else:
					tf.logging.info("==apply bi-directional LM model with bi-directional mask==")
					self.attention_mask = self.bi_attention_mask

				# Run the stacked transformer.
				# `sequence_output` shape = [batch_size, seq_length, hidden_size].

				if kargs.get('attention_type', 'normal_attention') == 'normal_attention':
					tf.logging.info("****** normal attention *******")
					transformer_model = bert_seq_modules.transformer_model
				elif kargs.get('attention_type', 'normal_attention') == 'rezero_transformer':
					transformer_model = bert_seq_modules.transformer_rezero_model
					tf.logging.info("****** rezero_transformer *******")
				else:
					tf.logging.info("****** normal attention *******")
					transformer_model = bert_seq_modules.transformer_model

				
				if kargs.get("reuse_mask", False):
					dropout_name = self.config.get("scope", "bert") + "/encoder"
					tf.logging.info("****** reuse mask: %s *******".format(dropout_name))
				else:
					dropout_name = None

				[self.all_encoder_layers,
				self.all_present,
				self.all_attention_scores,
				self.all_value_outputs] = transformer_model(
						input_tensor=self.embedding_output,
						attention_mask=self.attention_mask,
						hidden_size=self.config.hidden_size,
						num_hidden_layers=self.config.num_hidden_layers,
						num_attention_heads=self.config.num_attention_heads,
						intermediate_size=self.config.intermediate_size,
						intermediate_act_fn=bert_seq_modules.get_activation(self.config.hidden_act),
						hidden_dropout_prob=hidden_dropout_prob,
						attention_probs_dropout_prob=attention_probs_dropout_prob,
						initializer_range=self.config.initializer_range,
						do_return_all_layers=True,
						past=past,
						decode_loop_step=decode_loop_step,
						if_bp=if_bp,
						if_cache_decode=if_cache_decode,
						attention_fixed_size=self.config.get('attention_fixed_size', None),
						dropout_name=dropout_name)
				# self.cached_present = tf.stack(self.all_present, axis=1)

	def build_output_logits(self, **kargs):
		layer_num = kargs.get("layer_num", -1)
		self.sequence_output = self.get_encoder_layers(layer_num)
		input_shape_list = bert_utils.get_shape_list(self.sequence_output, expected_rank=3)
		batch_size = input_shape_list[0]
		seq_length = input_shape_list[1]
		hidden_dims = input_shape_list[2]

		embedding_projection = kargs.get('embedding_projection', None)

		scope = kargs.get('scope', None)
		if scope:
			scope = scope + '/' + 'cls/predictions'
		else:
			scope = 'cls/predictions'

		tf.logging.info("**** mlm generator scope **** %s", str(scope))

		# with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			if self.config.get('ln_type', 'postln') == 'preln':
				input_tensor = bert_modules.layer_norm(self.sequence_output)
				tf.logging.info("**** pre ln doing layer norm ****")
			elif self.config.get('ln_type', 'postln') == 'postln':
				input_tensor = self.sequence_output
				tf.logging.info("**** post ln ****")
			else:
				input_tensor = self.sequence_output
				tf.logging.info("**** post ln ****")

			# if config.get("embedding", "factorized") == "factorized":
			# 	projection_width = config.hidden_size
			# else:
			# 	projection_width = config.embedding_size

			if self.config.get("embedding", "none_factorized") == "none_factorized":
				projection_width = self.config.hidden_size
				tf.logging.info("==not using embedding factorized==")
			else:
				projection_width = self.config.get('embedding_size', self.config.hidden_size)
				tf.logging.info("==using embedding factorized: embedding size: %s==", str(projection_width))

			with tf.variable_scope("transform"):
				input_tensor = tf.layers.dense(
						input_tensor,
						units=projection_width,
						activation=bert_modules.get_activation(self.config.hidden_act),
						kernel_initializer=bert_modules.create_initializer(
								self.config.initializer_range))

				if self.config.get('ln_type', 'postln') == 'preln':
					input_tensor = input_tensor
					tf.logging.info("**** pre ln ****")
				elif self.config.get('ln_type', 'postln') == 'postln':
					input_tensor = bert_modules.layer_norm(input_tensor)
					tf.logging.info("**** post ln doing layer norm ****")
				else:
					input_tensor = bert_modules.layer_norm(input_tensor)
					tf.logging.info("**** post ln doing layer norm ****")

			if embedding_projection is not None:
				# batch x seq x hidden, embedding x hidden
				print(input_tensor.get_shape(), embedding_projection.get_shape())
				input_tensor = tf.einsum("abc,dc->abd", input_tensor, embedding_projection)
			else:
				print("==no need for embedding projection==")
				input_tensor = input_tensor

			output_bias = tf.get_variable(
					"output_bias",
					shape=[self.config.vocab_size],
					initializer=tf.zeros_initializer())
			# batch x seq x embedding
			logits = tf.einsum("abc,dc->abd", input_tensor, self.embedding_table)
			self.logits = tf.nn.bias_add(logits, output_bias)


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
				output_shape = bert_utils.get_shape_list(self.sequence_output[:, 0:1, :], expected_rank=[2,3])
				print(output_shape, bert_utils.get_shape_list(self.sequence_output, expected_rank=[2,3]))
				# if len(self.sequence_output[:, 0:1, :]) =
				first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
				self.pooled_output = tf.layers.dense(
						first_token_tensor,
						self.config.hidden_size,
						activation=tf.tanh,
						kernel_initializer=bert_modules.create_initializer(self.config.initializer_range))
	
	def get_present(self):
		return tf.stack(self.all_present, axis=1) # cache internal states
	
	def get_multihead_attention(self):
		return self.all_attention_scores
	
	def get_pooled_output(self):
		return self.pooled_output

	def get_value_layer(self):
		return self.all_value_outputs

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

	def get_embedding_output(self):
		return self.embedding_output_word

	def get_encoder_layers(self, layer_num):
		if layer_num >= 0 and layer_num <= len(self.all_encoder_layers) - 1:
			print("==get encoder layer==", layer_num)
			return self.all_encoder_layers[layer_num]
		else:
			return self.all_encoder_layers[-1]

	def get_sequence_output_logits(self):
		return self.logits

	def get_attention_mask(self):
		return self.attention_mask

