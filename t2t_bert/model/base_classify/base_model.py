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

		self.emb_mat = integration_func.generate_embedding_mat_v1(self.vocab_size, emb_len=self.emb_size,
									 init_mat=self.token_emb_mat, 
									 extra_symbol=self.extra_symbol, 
									 scope=self.scope+'_token_embedding')
		if self.config.with_char:
			self.char_mat = integration_func.generate_embedding_mat_v1(self.vocab_size, emb_len=self.emb_size,
									 init_mat=self.token_emb_mat, 
									 extra_symbol=self.extra_symbol, 
									 scope=self.scope+'_char_embedding')

	def build_char_embedding(self, input_char_ids, is_training, **kargs):

		input_char_mask = tf.cast(input_char_ids, tf.bool)
		input_char_len = tf.reduce_sum(tf.cast(input_char_mask, tf.int32), -1)

		reuse = kargs["reuse"]
		if self.config.char_embedding == "lstm":
			char_emb = char_embedding_utils.lstm_char_embedding(input_char_ids, input_char_len, self.char_mat, 
							self.config, is_training, reuse)
		elif self.config.char_embedding == "conv":
			char_emb = char_embedding_utils.conv_char_embedding(input_char_ids, input_char_len, self.char_mat, 
							self.config, is_training, reuse)
		return char_emb

	def build_word_embedding(self, input_ids):
		word_emb = tf.nn.embedding_lookup(self.emb_mat, input_ids)
		return word_emb

	def build_emebdder(self, input_ids, input_char_ids, is_training, **kargs):

		reuse = kargs["reuse"]
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb = self.build_emebdder(input_ids)
		if self.config.with_char:
			char_emb = self.build_char_embedding(input_char_ids, is_training, **kargs)
			self.word_emb = tf.concat([word_emb, char_emb], axis=-1)
		else:
			self.word_emb = word_emb

	def get_pooled_output(self):
		raise NotImplementedError

	# def build_predictor(self, features, is_training, labels, **kargs):
	# 	reuse = kargs["reuse"]
	# 	num_classes = self.config.num_classes
	# 	dropout_rate = tf.cond(is_training, 
	# 						lambda:self.config.dropout_rate,
	# 						lambda:0.0)

	# 	with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):

	# 		features = tf.nn.dropout(features, (1 - dropout_rate))
	# 		self.logits = tf.layers.dense(features, num_classes, use_bias=False)

	# 		if self.config.loss_type == "cross_entropy":
	# 			predictions = tf.nn.softmax(logits)
	# 			per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
 #                        		labels=labels)
	# 		elif self.config.loss_type == "focal_loss_multi_v1":
	# 			per_sample_loss, predictions = focal_loss_multi_v1(self.config, logits, labels)
	# 		elif self.config.loss_type == "center_loss_v2":
	# 			predictions = tf.nn.softmax(logits)
	# 			closs, centers = center_loss_v2(self.config, features, labels, centers=None)
	# 		loss = tf.reduce_mean(per_sample_loss)
	# 		return loss, per_sample_loss, predictions



