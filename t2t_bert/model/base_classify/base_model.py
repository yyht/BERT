import tensorflow as tf
import numpy as np

from utils.embed import integration_func
from loss.loss_utils import focal_loss_multi_v1, center_loss_v2

class BaseModel(object):
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
		self.extra_symbol = self.config.get("extra_symbol", None)
		self.emb_dropout_count = 0

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

	def build_word_embedding(self, input_ids, is_training=False, **kargs):

		self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_token_embedding',
                                     reuse=kargs.get("reuse", None),
                                     trainable=not self.config.get('use_pretrained', True))
		
		if is_training:
			dropout_rate = self.config.dropout_rate
		else:
			dropout_rate = 0.0

		if self.config.get('embedding_dropout', False) and is_training:
			embedding_matrix = tf.nn.dropout(self.emb_mat, 
										keep_prob=1-self.config.get('embedding_dropout', 0.01), 
										noise_shape=[self.vocab_size,1])
			tf.logging.info("***** word drop out *****")
		else:
			embedding_matrix = self.emb_mat
			tf.logging.info("***** none word drop *****")

		word_emb = tf.nn.embedding_lookup(embedding_matrix, input_ids)
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

	def build_emebdder(self, input_ids, input_char_ids, is_training, **kargs):

		reuse = kargs["reuse"]
		if is_training:
			dropout_rate = self.config.dropout_rate
		else:
			dropout_rate = 0.0
			
		# dropout_rate = tf.cond(is_training, 
		# 					lambda:self.config.dropout_rate,
		# 					lambda:0.0)

		word_emb = self.build_word_embedding(input_ids, is_training, **kargs)
		if self.config.with_char == "char":
			char_emb = self.build_char_embedding(input_char_ids, is_training, **kargs)
			self.word_emb = tf.concat([word_emb, char_emb], axis=-1)
		else:
			self.word_emb = word_emb

	def get_pooled_output(self):
		raise NotImplementedError

	def get_pooled_feature(self):
		raise NotImplementedError



