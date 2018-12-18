from utils.rnn import rnn_utils
from utils.bert import bert_utils

import tensorflow as tf
import numpy as np

class RNN(object):
	"""
	default scope: bert
	"""
	def __init__(self, config, *args, **kargs):
		self.config = copy.deepcopy(config)
		tf.logging.info(" begin to build {}".format(self.config.get("scope", "bert")))

	def build_encoder(self, input_ids, input_mask,
					dropout_rate, scope,
					**kargs):

		input_shape = bert_utils.get_shape_list(input_ids, expected_rank=3)
		batch_size = input_shape[0]
		input_size = input_shape[-1]
		reuse = kargs["reuse"]
		
		with tf.variable_scope(scope, reuse=reuse):
			if self.config.get("use_cudnn", True) == True:
				if self.config.get("rnn_dircetion", "bi"):
					rnn = rnn_utils.BiCudnnRNN(self.config["rnn_size"], 
						batch_size, input_size,
						num_layers=self.config["num_layers"], 
						dropout=dropout_rate, 
						kernel=self.get("rnn_type", "lstm"))
				else:
					rnn = rnn_utils.CudnnRNN(self.config["rnn_size"], 
						batch_size, input_size,
						num_layers=self.config["num_layers"], 
						dropout=dropout_rate, 
						kernel=self.get("rnn_type", "lstm"))

				input_lengths = tf.reduce_sum(input_mask, axis=-1)
				res, _ , _ = rnn(input_reps, 
						 seq_len=tf.cast(input_lengths, tf.int32), 
						 batch_first=True,
						 scope="rnn",
						 reuse=reuse)
				f_rep = res[:, :, 0:self.config["rnn_size"]]
				b_rep = res[:, :, self.config["rnn_size"]:2*self.config["rnn_size"]]
				outputs = res

				return [f_rep, b_rep, outputs]






