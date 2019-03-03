import tensorflow as tf
import numpy as np

from utils.textcnn import textcnn_utils
from utile.embed import integration_func
from model.base_classify import base_model

class TextCNN(base_model.BaseModel):
	def __init__(self, config):
		self.super(TextCNN, self).__init__(config)

	def build_encoder(self, input_ids, input_char_ids, is_training, **kargs):
		reuse = kargs["reuse"]
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb_dropout = tf.nn.dropout(self.word_emb, 1-dropout_rate)
		with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
			input_dim = word_emb_dropout.get_shape()[-1]
			sent_repres = match_utils.multi_highway_layer(word_emb_dropout, input_dim, self.config.highway_layer_num)

		input_mask = tf.cast(input_ids, tf.bool)
		input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

		mask = tf.expand_dims(input_mask, -1)
		sent_repres *= tf.cast(mask, tf.float32)

		with tf.variable_scope(self.config.scope+"_encoder", reuse=reuse):
			self.output = textcnn_utils.text_cnn(sent_repres, self.config.get("filter_size", [1,3,5,7]), 
					"textcnn", 
					self.emb_size, 
					self.config.num_filters, 
					max_pool_size=self.config.max_pool_size)
			print("output shape====", output.get_shape())

	def get_pooled_output(self):
		return self.output

	

