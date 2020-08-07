import tensorflow as tf
import numpy as np

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import numpy as np
import os

from utils.funnel_transformer import funnel_transformer_ops_v1 as funnel_transformer_ops
from utils.funnel_transformer import funnel_transformer_utils_v1 as funnel_transformer_utils

def input_embedding(net_config, initializer, inputs, is_training, seg_id=None, pos_id=None,
											word_embed_table=None, use_tpu=False, scope="input",
											reuse=tf.AUTO_REUSE, dtype=tf.float32,
											name='embed',
										embedding_table_adv=None,
										embedding_seq_adv=None,
										emb_adv_pos='emb_adv_pre',
										stop_gradient=False):
	"""Turn input ids to input embedding."""

	net_config = net_config
	ret_dict = {}

	##### Embedding
	def embed_func(x, pos_id, seg_id,
						initializer,
						embedding_table_adv, 
						embedding_seq_adv,
						emb_adv_pos, 
						stop_gradient):
		"""Word embed + Position embed + Segment embed (if provided)."""
		# Word embedding
		embed, word_embed_table = funnel_transformer_ops.embedding_lookup(
				x=x,
				n_embed=net_config.vocab_size,
				d_embed=net_config.d_embed,
				initializer=initializer,
				use_tpu=use_tpu,
				dtype=dtype,
				scope="word_embedding",
				embedding_table_adv=embedding_table_adv)

		if embedding_seq_adv is not None and emb_adv_pos == "emb_adv_pre":
			embedding_seq_adv = tf.cast(embedding_seq_adv, dtype=embed.dtype)
			if not stop_gradient:
				embed += embedding_seq_adv
				tf.logging.info("****** embed pre-processor with bp *******" )
			else:
				embedding_seq_adv += tf.stop_gradient(embed) - embed
				embed += embedding_seq_adv
				tf.logging.info("****** embedding_output_word pre-processor without bp *******" )

		if net_config.rel_attn_type == "null":
			# Position embedding
			if pos_id is None:
				pos_id = tf.cast(tf.range(tf.shape(x)[-1]), x.dtype)
			pos_emb, _ = funnel_transformer_ops.embedding_lookup(
					x=pos_id,
					n_embed=512,
					d_embed=net_config.d_embed,
					initializer=initializer,
					use_tpu=use_tpu,
					dtype=dtype,
					scope="position_embedding")
			embed += pos_emb

			# Segment embedding
			if seg_id is not None:
				seg_emb, _ = funnel_transformer_ops.embedding_lookup(
						x=seg_id % 2,
						n_embed=2,
						d_embed=net_config.d_embed,
						initializer=initializer,
						use_tpu=use_tpu,
						dtype=dtype,
						scope="segment_embedding")
				embed += seg_emb

		return embed, word_embed_table

	with tf.variable_scope(scope, reuse=reuse):
		##### Input embedding layer normalization and dropout
		word_emb, word_embed_table = embed_func(x=inputs,
																						pos_id=pos_id,
																						seg_id=seg_id,
																						initializer=initializer,
																						embedding_table_adv=embedding_table_adv,
																						embedding_seq_adv=embedding_seq_adv,
																						emb_adv_pos=emb_adv_pos,
																						stop_gradient=stop_gradient)
		word_emb = funnel_transformer_ops.layer_norm_op(word_emb, norm_shape=[net_config.d_embed])

		if embedding_seq_adv is not None and emb_adv_pos == "emb_adv_post":
			embedding_seq_adv = tf.cast(embedding_seq_adv, dtype=word_emb.dtype)
			if not stop_gradient:
				word_emb += embedding_seq_adv
				tf.logging.info("****** word_emb post-processor with bp *******" )
			else:
				embedding_seq_adv += tf.stop_gradient(word_emb) - word_emb
				word_emb += embedding_seq_adv
				tf.logging.info("****** word_emb post-processor without bp *******" )

		output = funnel_transformer_ops.dropout_op(word_emb,
														net_config.dropout,
														training=is_training,
														name=name)

	return output, word_embed_table, ret_dict

def encoder(net_config, 
						input_embed,
						is_training,
						initializer,
						seg_id=None,
						pos_id=None,
						input_mask=None,
						scope="encoder",
						reuse=tf.AUTO_REUSE,
						seq_type=None,
						mask_type=None,
						attn_structures=None,
						**kargs):
	"""Encoder of the Funnel-Transformer."""
	net_config = net_config
	ret_dict = {}

	with tf.variable_scope(scope, reuse=reuse):

		##### Input projection
		output, _ = funnel_transformer_utils.input_projection(net_config,
																												input_embed,
																												initializer)

		##### Encoder layers
		hiddens = []
		layer_dict = {}
		for block_idx in range(net_config.n_block):
			# prepare structures for relative attention
			if block_idx == 0:
				attn_structures_name = os.path.join(scope, str(block_idx), 'attn_structures')
				pos_enc, seg_mat, func_mask = funnel_transformer_utils.init_attn_structures(
						net_config,
						attn_structures,
						output, seg_id, pos_id, is_training, 
						attn_structures_name)
				if attn_structures is None:
					attn_structures = (pos_enc, seg_mat, func_mask)
			else:
				pre_attn_pooling_name = os.path.join(scope, str(block_idx), 'pre_attn_pooling')
				pool_ret = funnel_transformer_utils.pre_attn_pooling(
						net_config,
						output, pos_enc, seg_mat, input_mask, func_mask, block_idx,
						is_training, pre_attn_pooling_name)

				pooled_out, pos_enc, seg_mat, input_mask, func_mask = pool_ret
			attn_mask = None if input_mask is None else input_mask[:, None, None]

			for param_idx in range(net_config.block_param_size[block_idx]):
				##### current layer idx
				layer_idx = sum(net_config.block_param_size[:block_idx]) + param_idx
				with tf.variable_scope("layer_{}".format(layer_idx), reuse=reuse):
					cur_repeat_size = net_config.block_repeat_size[block_idx]
					for repeat_idx in range(cur_repeat_size):
						sub_idx = (param_idx * cur_repeat_size + repeat_idx)
						do_pooling = block_idx > 0 and sub_idx == 0
						print(do_pooling, "===do ppoling===")
						# prepare inputs to the current layer
						if do_pooling:
							if net_config.pool_q_only:
								q = pooled_out
								k = v = output
							else:
								q = k = v = pooled_out
						else:
							q = k = v = output

						# attention layer
						tfmxl_name = os.path.join(scope, str(block_idx), str(layer_idx), str(repeat_idx), 'tfmxl_layer')
						output, layer_dict = funnel_transformer_utils.tfmxl_layer(
								net_config=net_config,
								q=q,
								k=k,
								v=v,
								pos_enc=pos_enc,
								seg_mat=seg_mat,
								attn_mask=attn_mask,
								is_training=is_training,
								initializer=initializer,
								func_mask=func_mask,
								name=tfmxl_name)

						# post-attention pooling
						if do_pooling:
							post_attn_pooling_name = os.path.join(scope, str(block_idx), str(layer_idx), str(repeat_idx), 'post_attn_pooling')
							pool_ret = funnel_transformer_utils.post_attn_pooling(
									net_config,
									pos_enc, seg_mat, input_mask, func_mask, block_idx,
									is_training, 
									post_attn_pooling_name)
							pos_enc, seg_mat, input_mask, func_mask = pool_ret
							attn_mask = None if input_mask is None \
									else input_mask[:, None, None]

						# update ret dict
						hiddens.append(output)
						prefix = "block_{}/layer_{}/repeat_{}".format(
								block_idx, layer_idx, repeat_idx)
						funnel_transformer_ops.update_ret_dict(ret_dict, layer_dict, prefix)

	return output, hiddens, ret_dict, attn_structures

def decoder(net_config, 
						hiddens,
						is_training,
						initializer,
						input_mask=None,
						seg_id=None,
						pos_id=None,
						scope="decoder",
						reuse=tf.AUTO_REUSE,
						attn_structures=None,
						**kargs):
	"""Decode a compressed sequence into a full sequence."""
	net_config = net_config
	ret_dict = {}

	output, bridge_dict = funnel_transformer_utils.bridge_layer(
			net_config,
			hiddens, input_mask, reuse=reuse)
	funnel_transformer_ops.update_ret_dict(ret_dict, bridge_dict, "bridge")

	if net_config.decoder_depth == 0:
		return output, ret_dict

	# prepare structures for relative attention
	attn_structures_name = os.path.join(scope, 'attn_structures')
	# pos_enc, seg_mat, func_mask = funnel_transformer_utils.init_attn_structures(
	# 		output, seg_id, pos_id, is_training)

	pos_enc, seg_mat, func_mask = funnel_transformer_utils.init_attn_structures(
						net_config,
						attn_structures,
						output, seg_id, pos_id, is_training, 
						attn_structures_name)
	attn_mask = None if input_mask is None else input_mask[:, None, None]
	print("==decoder attn mask==", attn_mask)

	# Decoder layers
	n_enc_param_layer = sum(net_config.block_param_size)
	with tf.variable_scope(scope, reuse=reuse):
		for param_idx in range(net_config.decoder_param_size):
			layer_idx = n_enc_param_layer + param_idx
			with tf.variable_scope("layer_{}".format(layer_idx), reuse=reuse):
				for repeat_idx in range(net_config.decoder_repeat_size):
					tfmxl_name = os.path.join(scope, str(layer_idx), str(repeat_idx), 'tfmxl_layer')
					output, layer_dict = funnel_transformer_utils.tfmxl_layer(
							net_config=net_config,
							q=output,
							k=output,
							v=output,
							pos_enc=pos_enc,
							seg_mat=seg_mat,
							attn_mask=attn_mask,
							is_training=is_training,
							initializer=initializer,
							func_mask=func_mask,
							name=tfmxl_name)

					funnel_transformer_ops.update_ret_dict(
							ret_dict, layer_dict,
							"layer_{}/repeat_{}".format(layer_idx, repeat_idx))

	return output, ret_dict