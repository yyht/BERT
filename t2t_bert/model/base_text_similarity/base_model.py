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

	def build_emebdder(self, input_ids, input_char_ids, is_training, **kargs):

		reuse = kargs["reuse"]
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb = self.build_word_embedding(input_ids, **kargs)
		if self.config.with_char == "char":
			char_emb = self.build_char_embedding(input_char_ids, is_training, **kargs)
			self.word_emb = tf.concat([word_emb, char_emb], axis=-1)
		else:
			self.word_emb = word_emb

	def get_pooled_output(self):
		raise NotImplementedError