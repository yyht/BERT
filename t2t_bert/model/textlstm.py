import tensorflow as tf
import numpy as np

from utils.bimpm import match_utils
from utils.embed import integration_func
from model.base_classify import base_model

import tensorflow as tf
from utils.bimpm import layer_utils, match_utils
from utils.qanet import qanet_layers
from utils.esim import esim_utils
from utils.slstm import slstm_utils
from utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from utils.label_network import label_network_utils

EPSILON = 1e-8

class TextLSTM(base_model.BaseModel):
	def __init__(self, config):
		super(TextLSTM, self).__init__(config)
		
	def build_encoder(self, input_ids, input_char_ids, is_training, **kargs):

		reuse = kargs.get("reuse", None)
		dropout_rate = tf.cond(is_training, 
							lambda:self.config.dropout_rate,
							lambda:0.0)

		word_emb = tf.nn.dropout(self.word_emb, 1 - dropout_rate)
		with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):

			input_dim = word_emb.get_shape()[-1]
			input_mask = tf.cast(input_ids, tf.bool)
			input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

			sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
			
			if self.config.rnn == "lstm":
				[sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
								self.config.context_lstm_dim, 
								input_lengths=input_len, 
								scope_name=self.config.scope, 
								reuse=reuse, 
								is_training=is_training,
								dropout_rate=dropout_rate, 
								use_cudnn=self.config.use_cudnn)

			elif self.config.rnn == "slstm":

				word_emb_proj = tf.layers.dense(word_emb, 
										self.config.slstm_hidden_size)

				initial_hidden_states = word_emb_proj
				initial_cell_states = tf.identity(initial_hidden_states)

				[new_hidden_states, 
				new_cell_states, 
				dummynode_hidden_states] = slstm_utils.slstm_cell(self.config, 
									self.config.scope, 
									self.config.slstm_hidden_size, 
									input_len, 
									initial_hidden_states, 
									initial_cell_states, 
									self.config.slstm_layer_num,
									dropout_rate, reuse=reuse)

				sent_repres = new_hidden_states

			if self.config.multi_head:
				mask = tf.cast(input_mask, tf.float32)
				ignore_padding = (1 - mask)
				ignore_padding = label_network_utils.attention_bias_ignore_padding(ignore_padding)
				encoder_self_attention_bias = ignore_padding

				sent_repres = label_network_utils.multihead_attention_texar(
					sent_repres, 
					memory=None, 
					memory_attention_bias=encoder_self_attention_bias,
					num_heads=8, 
					num_units=128, 
					dropout_rate=dropout_rate, 
					scope="multihead_attention")

			v_attn = self_attn.multi_dimensional_attention(
				sent_repres, input_mask,  self.config.scope+'_multi_dim_attn',
				1 - dropout_rate, is_training, self.config.weight_decay, "relu")
			
			mask = tf.expand_dims(input_mask, -1)
			v_sum = tf.reduce_sum(sent_repres*tf.cast(mask, tf.float32), 1)
			v_ave = tf.div(v_sum, tf.expand_dims(tf.cast(input_lengths, tf.float32)+EPSILON, -1))

			v_max = tf.reduce_max(qanet_layers.mask_logits(sent_repres, mask), axis=1)

			v_last = esim_utils.last_relevant_output(sent_repres, input_lengths)

			self.output = tf.concat([v_ave, v_max, v_last, v_attn], axis=-1)

	def get_pooled_output(self):
		return self.output

		