from model.dsmm import dsmm
import tensorflow as tf
import numpy as np
from utils.dsmm.tf_common.nn_module import encode, attend, mlp_layer
from utils.textcnn import textcnn_utils
from utils.bimpm import match_utils
from model.match_pyramid import mp_cnn

class MatchPyramid(dsmm.DSMM):
	def __init__(self, config):
		super(MatchPyramid, self).__init__(config)

	def _encode(self, input_ids, input_char_ids,
				is_training, **kargs):

		reuse = kargs.get("reuse", None)

		with tf.variable_scope(self.config.scope+"_semantic_encode", reuse=reuse):

			emb_seq = self._embd_seq(input_ids, input_char_ids, is_training, reuse=reuse)

			input_dim = emb_seq.shape[-1].value

			input_mask = tf.cast(input_ids, tf.bool)
			input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)

			enc_seq = encode(seq_input, method=self.config["encode_method"],
								 input_dim=input_dim,
								 params=self.config,
								 sequence_length=input_len,
								 mask_zero=self.config["embedding_mask_zero"],
								 scope_name=self.scope + "enc_seq", 
								 reuse=reuse,
								 training=is_training)

		return emb_seq, enc_seq

	def _semantic_encode(self, input_ids_a, input_char_ids_a, 
				input_ids_b, input_char_ids_b,
				is_training, **kargs):

		emb_seq_a, enc_seq_a = self._encode(input_ids_a, 
												input_char_ids_a, 
												is_training,
												reuse=kargs.get("reuse", None))
		emb_seq_b, enc_seq_b = self._encode(input_ids_b,
												input_char_ids_b,
												is_training,
												reuse=True)

		return emb_seq_a, enc_seq_a, emb_seq_b, enc_seq_b

	def _semantic_interaction(self, input_ids_a, input_char_ids_a, 
				input_ids_b, input_char_ids_b,
				emb_seq_a, enc_seq_a, emb_seq_b, enc_seq_b,
				is_training, **kargs):

		emb_match_matrix_dot_product = tf.einsum("abd,acd->abc", emb_seq_a, emb_seq_b)
		emb_match_matrix_dot_product = tf.expand_dims(emb_match_matrix_dot_product, axis=-1) # batch x seq_len_a x seq_len_b x 1

		match_matrix_identity = tf.expand_dims(tf.cast(
			tf.equal(
				tf.expand_dims(input_ids_a, 2),
				tf.expand_dims(input_ids_b, 1)
			), tf.float32), axis=-1) # batch x seq_len_a x seq_len_b x 1

		emb_match_matrix_element_product = tf.expand_dims(emb_seq_a, 2) * tf.expand_dims(
			emb_seq_b, 1)

		enc_match_matrix_dot_product = tf.expand_dims(
			tf.einsum("abd,acd->abc", enc_seq_a, enc_seq_b), axis=-1)
		enc_match_matrix_element_product = tf.expand_dims(enc_seq_a, 2) * tf.expand_dims(
			enc_seq_b, 1)

		match_matrix = tf.concat([
			emb_match_matrix_dot_product,
			match_matrix_identity,
			emb_match_matrix_element_product,
			enc_match_matrix_dot_product,
			enc_match_matrix_element_product
		], axis=-1)

		return match_matrix

	def _semantic_aggerate(self, match_matrix,
				is_training, **kargs):

		self.aggerate_feature = mp_cnn._mp_semantic_feature_layer(match_matrix, kargs.get("dpool_index", None))

	def get_pooled_output(self):
		return self.aggerate_feature


