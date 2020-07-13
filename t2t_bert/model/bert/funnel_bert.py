import numpy as np
import tensorflow as tf
import copy
from utils.funnel_transformer import funnel_transformer_modules
from utils.funnel_transformer import funnel_transformer_utils
from utils.funnel_transformer import funnel_transformer_ops

import numpy as np
import tensorflow as tf
import copy
import os

def parse_depth_string(depth_str):
		depth_config = depth_str.split("x")
		if len(depth_config) == 1:
			depth_config.append(1)
		assert len(depth_config) == 2, "Require two-element depth config."

		return list(map(int, depth_config))

def get_initializer(net_config):
		"""Get variable intializer."""
		net_config = net_config
		if net_config.init == "uniform":
			initializer = tf.initializers.random_uniform(
					minval=-net_config.init_range,
					maxval=net_config.init_range,
					seed=None)
		elif net_config.init == "normal":
			initializer = tf.initializers.random_normal(
					stddev=net_config.init_std,
					seed=None)
		elif net_config.init == "truncated_normal":
			initializer = tf.initializers.truncated_normal(
					stddev=net_config.init_std,
					seed=None)
		else:
			raise ValueError("Initializer {} not supported".format(net_config.init))
		return initializer

class FunnelTransformer(object):
	"""
	default scope: bert
	"""
	def __init__(self, config, *args, **kargs):
		self.config = copy.deepcopy(config)
		tf.logging.info(" begin to build {}".format(self.config.get("scope", "bert")))
		self.ret_dict = {}
		self.block_size = self.config.block_size
		self.block_depth = []
		self.block_param_size = []
		self.block_repeat_size = []
		for cur_block_size in self.block_size.split("_"):
			cur_block_size = parse_depth_string(cur_block_size)
			self.block_depth.append(cur_block_size[0] * cur_block_size[1])
			self.block_param_size.append(cur_block_size[0])
			self.block_repeat_size.append(cur_block_size[1])
		self.n_block = len(self.block_depth)
		self.config.initializer_range = self.config.init_std

		# assert not (self.n_block == 1 and decoder_size != "0"), \
		# 		"Models with only 1 block does NOT need a decoder."
		self.decoder_size = self.config.decoder_size
		decoder_size = parse_depth_string(self.decoder_size)
		self.decoder_depth = decoder_size[0] * decoder_size[1]
		self.decoder_param_size = decoder_size[0]
		self.decoder_repeat_size = decoder_size[1]

		self.config.n_block = self.n_block
		self.config.block_depth = self.block_depth
		self.config.block_param_size = self.block_param_size
		self.config.block_repeat_size = self.block_repeat_size

		self.config.decoder_depth = decoder_size[0] * decoder_size[1]
		self.config.decoder_param_size = decoder_size[0]
		self.config.decoder_repeat_size = decoder_size[1]
		self.attn_structures = None
		self.initializer_range = self.config.init_std

	def build_embedder(self, input_ids, token_type_ids, 
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									use_bfloat16,
									is_training,
									use_tpu,
									**kargs):

		embedding_table_adv = kargs.get('embedding_table_adv', None)
		print(embedding_table_adv, "==embedding-adv")

		embedding_seq_adv = kargs.get('embedding_seq_adv', None)
		print(embedding_seq_adv, "==embedding-adv")

		emb_adv_pos = kargs.get("emb_adv_pos", "emb_adv_post")
		stop_gradient = kargs.get("stop_gradient", False)

		if self.config.get('embedding_scope', None):
			embedding_scope = self.config['embedding_scope']
			tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s ==", embedding_scope)
		else:
			embedding_scope = self.config.get("scope", "model")
			tf.logging.info("==using embedding scope of original model_config.embedding_scope: %s ==", embedding_scope)

		initializer = get_initializer(self.config)

		dtype = tf.float32 if not use_bfloat16 else tf.bfloat16
		with tf.variable_scope(embedding_scope, reuse=tf.AUTO_REUSE):
			embed_name = os.path.join(embedding_scope, 'embed')
			[self.input_embed, 
			self.word_embed_table, 
			self.emb_dict] = funnel_transformer_modules.input_embedding(
				self.config,
				initializer,
				input_ids, 
				is_training, 
				seg_id=token_type_ids, 
				use_tpu=use_tpu, 
				dtype=dtype,
				embedding_table_adv=embedding_table_adv,
				embedding_seq_adv=embedding_seq_adv,
				emb_adv_pos=emb_adv_pos,
				stop_gradient=stop_gradient,
				name=embed_name)
			funnel_transformer_ops.update_ret_dict(self.ret_dict, self.emb_dict, "emb")

	def build_encoder(self, input_ids, input_mask, 
									token_type_ids,
									hidden_dropout_prob, 
									attention_probs_dropout_prob,
									is_training,
									use_bfloat16,
									embedding_output=None,
									**kargs):

		initializer = get_initializer(self.config)
		dtype = tf.float32 if not use_bfloat16 else tf.bfloat16
		scope = self.config.get("scope", "model")
		self.attn_structures = None

		if embedding_output is not None:
			embedding_seq_output = embedding_output
			tf.logging.info("****** outer-embedding_seq_output *******")
		else:
			embedding_seq_output = self.input_embed
			tf.logging.info("****** self-embedding_seq_output *******")

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			[self.encoder_output, 
				self.encoder_hiddens, 
				self.enc_dict,
				self.attn_structures] = funnel_transformer_modules.encoder(
					self.config,
					embedding_seq_output,
					is_training,
					initializer,
					seg_id=token_type_ids,
					input_mask=input_mask,
					attn_structures=self.attn_structures)
			print(self.attn_structures, "==attention structures==")

		funnel_transformer_ops.update_ret_dict(self.ret_dict, 
																					self.enc_dict, 
																					"enc")

	def build_decoder(self, hiddens, input_ids, input_mask, 
									token_type_ids, is_training, **kargs):
		# decoder
		if self.config.n_block > 1:
			initializer = get_initializer(self.config)
			scope = self.config.get("scope", "model")
			with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
				[self.decoder_output, 
				self.dec_dict] = funnel_transformer_modules.decoder(
						self.config,
						hiddens,
						input_mask=input_mask,
						seg_id=token_type_ids,
						is_training=is_training,
						initializer=initializer,
						attn_structures=self.attn_structures)
			funnel_transformer_ops.update_ret_dict(self.ret_dict, 
																						self.dec_dict, 
																						"dec")
		else:
			self.decoder_output = None
			self.dec_dict = {}

	def build_pooler(self, *args, **kargs):
		reuse = kargs["reuse"]
		layer_num = kargs.get("layer_num", -1)
		scope = self.config.get("scope", "model")
		with tf.variable_scope(scope, reuse=reuse):
			self.sequence_output = self.get_encoder_layers(layer_num)

			# The "pooler" converts the encoded sequence tensor of shape
			# [batch_size, seq_length, hidden_size] to a tensor of shape
			# [batch_size, hidden_size]. This is necessary for segment-level
			# (or segment-pair-level) classification tasks where we need a fixed
			# dimensional representation of the segment.
			with tf.variable_scope("pooler"):
				# We "pool" the model by simply taking the hidden state corresponding
				# to the first token. We assume that this has been pre-trained
				initializer = get_initializer(self.config)
				first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
				self.pooled_output = tf.layers.dense(
						first_token_tensor,
						self.config.hidden_size,
						activation=tf.tanh,
						kernel_initializer=initializer,
						use_bias=True)
	
	def get_multihead_attention(self):
		attention_scores_list = []
		for key in self.ret_dict:
			if 'enc' in key and 'attn_prob' in key:
				attention_scores_list.append(self.ret_dict[key])
		return attention_scores_list
	
	def get_pooled_output(self):
		return self.pooled_output

	def put_task_output(self, input_repres):
		self.task_repres = input_repres

	def get_task_output(self):
		return self.task_repres

	def get_value_layer(self):
		return None

	def get_embedding_projection_table(self):
		return None

	def get_sequence_output(self, output_type='encoder'):
		"""Gets final hidden layer of encoder.

		Returns:
			float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
			to the final hidden of the transformer encoder.
		"""
		if output_type == 'encoder':
			return self.encoder_output
		elif output_type == 'decoder':
			return self.decoder_output
		else:
			return self.encoder_output

	def get_all_encoder_layers(self, output_type='encoder'):
		if output_type == 'encoder':
			return self.encoder_hiddens
		elif output_type == 'decoder':
			return None
		else:
			return self.encoder_hiddens

	def get_embedding_table(self):
		return self.word_embed_table

	def get_embedding_output(self):
		return self.input_embed

	def get_encoder_layers(self, layer_num):
		if layer_num >= 0 and layer_num <= len(self.encoder_hiddens) - 1:
			print("==get encoder layer==", layer_num)
			return self.encoder_hiddens[layer_num]
		else:
			return self.encoder_hiddens[-1]
