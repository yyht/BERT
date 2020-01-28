import tensorflow as tf
import numpy as np
from model.gpt import gpt_utils

class GPT(object):
	def __init__(self, config, *args, **kargs):
		self.config = config

	def build_model(self, hparams, X, past=None, scope='model', reuse=False):
		with tf.variable_scope(scope, reuse=reuse):
			self.results = {}
			batch, sequence = gpt_utils.shape_list(X)

			wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
								 initializer=tf.random_normal_initializer(stddev=0.01))
			wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
								 initializer=tf.random_normal_initializer(stddev=0.02))
			past_length = 0 if past is None else tf.shape(past)[-2]
			h = tf.gather(wte, X) + tf.gather(wpe, gpt_utils.positions_for(X, past_length))

			# Transformer
			presents = []
			pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
			assert len(pasts) == hparams.n_layer
			for layer, past in enumerate(pasts):
				h, present = gpt_utils.block(h, 'h%d' % layer, past=past, hparams=hparams)
				presents.append(present)
			self.results['present'] = tf.stack(presents, axis=1) # cache internal states
			print(self.results['present'].get_shape())
			self.h = gpt_utils.norm(h, 'ln_f')

			# Language model loss.  Do tokens <n predict token n?
			h_flat = tf.reshape(self.h, [batch*sequence, hparams.n_embd])
			logits = tf.matmul(h_flat, wte, transpose_b=True)
			logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
			self.results['logits'] = logits
			
	def get_sequence_output_logits(self):
		return self.results['logits']

	def get_sequence_output_activation(self):
		return self.h

	def get_pooled_output(self):
		# output of last activation of last block of last valid tokens
		return self.h[:, -2]

	def get_present(self):
		return self.results['present']

