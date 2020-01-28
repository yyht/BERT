import tensorflow as tf
import numpy as np

from utils.textcnn import textcnn_utils
from utils.bimpm import match_utils
from utils.embed import integration_func
from model.base_classify import base_model

from utils.qanet.qanet_layers import highway

class DAN(base_model.BaseModel):
	def __init__(self, config):
		super(DAN, self).__init__(config)

	def build_encoder(self, input_ids, input_char_ids, is_training, **kargs):
		reuse = kargs["reuse"]
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb_dropout = tf.nn.dropout(self.word_emb, 1-dropout_rate)
		with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
			input_dim = word_emb_dropout.get_shape()[-1]
			if self.config.get("highway", "dense_highway") == "dense_highway":
				sent_repres = match_utils.multi_highway_layer(word_emb_dropout, input_dim, self.config.highway_layer_num)
			elif self.config.get("highway", "dense_highway") == "conv_highway":
				sent_repres = highway(word_emb_dropout, 
								size = self.config.num_filters, 
								scope = "highway", 
								dropout = dropout_rate, 
								reuse = None)
			else:
				sent_repres = word_emb_dropout

		input_mask = tf.cast(input_ids, tf.bool)
		input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

		mask = tf.expand_dims(input_mask, -1)
		sent_repres *= tf.cast(mask, tf.float32)

		with tf.variable_scope(self.config.scope+"_encoder", reuse=reuse):
			self.output = tf.reduce_sum(sent_repres, axis=1) / tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
			print("output shape====", self.output.get_shape())

	def get_pooled_output(self):
		return self.output

	

