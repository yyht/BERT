import tensorflow as tf
import numpy as np
from model.utils.qanet import qanet_layers

import math

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

def query_context_attention(query, context, max_query_len, max_context_len, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):
        # context_ = tf.transpose(context, [0,2,1])
        hiddem_dim = query.get_shape()[-1]

        query_ = tf.nn.l2_normalize(query, axis=-1)
        context_ = tf.nn.l2_normalize(context, axis=-1)

        # attn_W = tf.get_variable("AttnW", dtype=tf.float32,
        #                             shape=[hiddem_dim, hiddem_dim],
        #                             initializer=initializer)

        S = tf.matmul(query_, tf.transpose(context_, [0,2,1]))

        # S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

        mask_q = tf.expand_dims(query_mask, 1)
        mask_c = tf.expand_dims(context_mask, 1)

        S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
        c2q = tf.matmul(S_, context)

        S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        q2c = tf.matmul(S_T, query)

        query_attention_outputs = c2q #tf.concat([query*c2q, c2q], axis=-1)
        # query_attention_outputs *= tf.expand_dims(tf.cast(query_mask, tf.float32), -1)

        context_attention_outputs = q2c #tf.concat([context*q2c, q2c], axis=-1)
        # context_attention_outputs *= tf.expand_dims(tf.cast(context_mask, tf.float32), -1)

        # query_attention_outputs = tf.nn.dropout(query_attention_outputs, 1 - dropout_ratio)
        # context_attention_outputs = tf.nn.dropout(context_attention_outputs, 1 - dropout_ratio)

        return query_attention_outputs, context_attention_outputs

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if var.get_shape().ndims <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter