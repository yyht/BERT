import tensorflow as tf
import numpy as np
from utils.embed import integration_func

from utils.textcnn import textcnn_utils
from utils.bimpm import match_utils

import tensorflow as tf
import numpy as np

class DSMM(object):
	def __init__(self, config):
		self.config = config
		self.token_emb_mat = self.config["token_emb_mat"]
		self.char_emb_mat = self.config["char_emb_mat"]
		self.vocab_size = int(self.config["vocab_size"])
		self.char_vocab_size = int(self.config["char_vocab_size"])
		self.max_length = int(self.config["max_length"])
		self.emb_size = int(self.config["emb_size"])
		self.scope = self.config["scope"]
		self.char_dim = self.config.get("char_emb_size", 300)
		self.extra_symbol = self.config.get("extra_symbol", ["<pad>", "<unk>", "<s>", "</s>"])

	def build_char_embedding(self, input_char_ids, is_training, **kargs):

		input_char_mask = tf.cast(input_char_ids, tf.bool)
		input_char_len = tf.reduce_sum(tf.cast(input_char_mask, tf.int32), -1)
		reuse = kargs["reuse"]

		if self.config.with_char == "char":
			self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_char_embedding',
                                     reuse=kargs.get("reuse", None),
                                     trainable=kargs.get("trainable", False))

		if self.config.char_embedding == "lstm":
			char_emb = char_embedding_utils.lstm_char_embedding(input_char_ids, input_char_len, self.char_mat, 
							self.config, is_training, reuse)
		elif self.config.char_embedding == "conv":
			char_emb = char_embedding_utils.conv_char_embedding(input_char_ids, input_char_len, self.char_mat, 
							self.config, is_training, reuse)
		return char_emb

	def build_word_embedding(self, input_ids, **kargs):

		self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_token_embedding',
                                     reuse=kargs.get("reuse", None),
                                     trainable=False)
		word_emb = tf.nn.embedding_lookup(self.emb_mat, input_ids)
		if self.config.get("trainable_embedding", False):
			self.trainable_emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_token_embedding',
                                     reuse=kargs.get("reuse", None),
                                     trainable=True)
			trainable_word_emb = tf.nn.embedding_lookup(self.trainable_emb_mat, input_ids)
			word_emb = tf.concat([word_emb, trainable_word_emb], axis=-1)

		return word_emb

	def _embd_seq(self, input_ids, input_char_ids, is_training, **kargs):
		reuse = kargs["reuse"]
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb = self.build_word_embedding(input_ids, **kargs)
		if self.config.with_char == "char":
			char_emb = self.build_char_embedding(input_char_ids, is_training, **kargs)
			word_emb = tf.concat([word_emb, char_emb], axis=-1)
		else:
			word_emb = word_emb

		word_emb_dropout = tf.nn.dropout(word_emb, 1-dropout_rate)
		input_mask = tf.cast(input_ids, tf.bool)
		input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

		with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
			input_dim = word_emb_dropout.get_shape()[-1]
			seq_input = match_utils.multi_highway_layer(word_emb_dropout, input_dim, self.config.highway_layer_num)
			seq_input *= tf.cast(input_mask, tf.float32)

		return seq_input

	def _semantic_encode(self, input_ids_a, input_char_ids_a, 
			input_ids_b, input_char_ids_b, is_training, **kargs):
		pass

	def _semantic_interaction(self, input_ids_a, input_char_ids_a, 
			input_ids_b, input_char_ids_b, is_training, **kargs):
		pass

	def _semantic_aggerate(self, input_ids_a, input_char_ids_a, 
			input_ids_b, input_char_ids_b, is_training, **kargs):
		pass

	# def _semantic_feature_layer(self, input_ids, input_char_ids, is_training, **kargs):
	# 	reuse = kargs.get("reuse", None)

	# 	seq_input = self._embd_seq(input_ids, input_char_ids, is_training, **kargs)
	# 	input_dim = seq_input.shape[-1].value
	# 	input_mask = tf.cast(input_ids, tf.bool)
	# 	input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

	# 	with tf.variable_scope(self.config.scope+"_semantic_feature_layer", reuse=reuse)
	# 		enc_seq = encode(seq_input, method=self.config["encode_method"],
	# 						 input_dim=input_dim,
	# 						 params=self.config,
	# 						 sequence_length=input_len,
	# 						 mask_zero=self.config["embedding_mask_zero"],
	# 						 scope_name=self.scope + "enc_seq", 
	# 						 reuse=reuse,
	# 						 training=is_training)

	# 		#### attend
	# 		feature_dim = self.config["encode_dim"]
	# 		print("==semantic feature dim==", feature_dim, enc_seq.get_shape())
	# 		context = None

	# 		att_seq = attend(enc_seq, context=context,
	# 						 encode_dim=self.config["encode_dim"],
	# 						 feature_dim=feature_dim,
	# 						 attention_dim=self.config["attention_dim"],
	# 						 method=self.config["attend_method"],
	# 						 scope_name=self.scope + "att_seq",
	# 						 reuse=reuse, 
	# 						 num_heads=self.config["attention_num_heads"])
	# 		print("==semantic layer attention seq shape==", att_seq.get_shape())
	# 		#### MLP nonlinear projection
	# 		sem_seq = mlp_layer(att_seq, fc_type=self.config["fc_type"],
	# 							hidden_units=self.config["fc_hidden_units"],
	# 							dropouts=self.config["fc_dropouts"],
	# 							scope_name=self.scope + "sem_seq",
	# 							reuse=reuse,
	# 							training=self.is_training,
	# 							seed=self.config["random_seed"])
	# 		print("==semantic layer mlp seq shape==", sem_seq.get_shape())
	# 		return enc_seq, att_seq, sem_seq

	# def _interaction_semantic_feature_layer(self, input_ids_a, input_char_ids_a, 
	# 		input_ids_b, input_char_ids_b, is_training, **kargs):

	# 	seq_input_left = self._embd_seq(input_ids_a, input_char_ids_a, is_training, reuse=None)
	# 	seq_input_right = self._embd_seq(input_ids_b, input_char_ids_b, is_training, reuse=True)

	# 	#### encode
	# 	input_dim = seq_input_left.shape[-1].value
	# 	enc_seq_left = encode(seq_input_left, method=self.config["encode_method"],
	# 						  input_dim=input_dim,
	# 						  params=self.config,
	# 						  sequence_length=seq_len_left,
	# 						  mask_zero=self.config["embedding_mask_zero"],
	# 						  scope_name=self.config.scope + "enc_seq", reuse=False,
	# 						  training=is_training)
	# 	enc_seq_right = encode(seq_input_right, method=self.config["encode_method"],
	# 						   input_dim=input_dim,
	# 						   params=self.config,
	# 						   sequence_length=seq_len_right,
	# 						   mask_zero=self.config["embedding_mask_zero"],
	# 						   scope_name=self.config.scope + "enc_seq", reuse=True,
	# 						   training=is_training)

	# 	#### attend
	# 	# [batchsize, s1, s2]
	# 	att_mat = tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right)
	# 	feature_dim = self.config["encode_dim"] + self.config["max_seq_len"]
	# 	att_seq_left = attend(enc_seq_left, context=att_mat, feature_dim=feature_dim,
	# 							   method=self.config["attend_method"],
	# 							   scope_name=self.config.scope + "att_seq",
	# 							   reuse=False)
	# 	att_seq_right = attend(enc_seq_right, context=tf.transpose(att_mat), feature_dim=feature_dim,
	# 						  method=self.config["attend_method"],
	# 						  scope_name=self.scope + "att_seq",
	# 						  reuse=True)

	# 	#### MLP nonlinear projection
	# 	sem_seq_left = mlp_layer(att_seq_left, fc_type=self.config["fc_type"],
	# 							 hidden_units=self.config["fc_hidden_units"],
	# 							 dropouts=self.config["fc_dropouts"],
	# 							 scope_name=self.config.scope + "sem_seq",
	# 							 reuse=False,
	# 							 training=is_training,
	# 							 seed=self.config["random_seed"])
	# 	sem_seq_right = mlp_layer(att_seq_right, fc_type=self.config["fc_type"],
	# 							  hidden_units=self.config["fc_hidden_units"],
	# 							  dropouts=self.config["fc_dropouts"],
	# 							  scope_name=self.scope + "sem_seq",
	# 							  reuse=True,
	# 							  training=is_training,
	# 							  seed=self.config["random_seed"])

	# 	return enc_seq_left, att_seq_left, sem_seq_left, \
	# 			enc_seq_right, att_seq_right, sem_seq_right
