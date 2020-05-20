import tensorflow as tf
import numpy as np

from utils.qanet import qanet_layers
from utils.bert import bert_utils

def residual_conv_encoder(x, input_mask, dropout, input_projection,
						config,
						is_training=False,
						):

	if is_training:
		dropout_rate = 0.1
	else:
		dropout_rate = 0.0

	in_val_shape = input_shape = bert_utils.get_shape_list(x, expected_rank=[3])

	seq_output = qanet_layers.residual_block(x,
                num_blocks = config.get('num_blocks', 1),
                num_conv_layers = config.get('num_conv_layers', 2),
                kernel_size = 7,
                mask = input_mask,
                num_filters = config.get('num_filters', 64),
                num_heads = config.get('num_heads', 1),
                seq_len = in_val_shape[-1],
                scope = "Encoder_Residual_Block",
                reuse = tf.AUTO_REUSE,
                bias = False,
                input_projection = input_projection,
                dropout = dropout)
	return seq_output