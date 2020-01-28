import tensorflow as tf
import numpy as np
from model.utils.qanet import qanet_layers

import math

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)


def bilinear_attention(query, context, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):
        context_ = tf.transpose(context, [0,2,1])
        hidden_dim = query.get_shape()[-1]

        attn_W = tf.get_variable("AttnW", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

        weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]])

        S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

        mask_q = tf.expand_dims(query_mask, 1)
        mask_c = tf.expand_dims(context_mask, 1)

        S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
        c2q = tf.matmul(S_, context)

        S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        q2c = tf.matmul(S_T, query)

        return c2q, q2c


def concat_attention(query, context, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    hidden_dim = query.get_shape()[-1]
    Wc_1 = tf.get_variable("Wc_1", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Wc_2 = tf.get_variable("Wc_2", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Vc = tf.get_variable("Vc", dtype=tf.float32,
                                    shape=[hidden_dim, 1],
                                    initializer=initializer)

    # batch x len x hidden_dim
    attention_1 = tf.einsum("abc,cd->abd", query, Wc_1)
    attention_2 = tf.einsum("abc,cd->abd", context, Wc_2)

    # concat attention
    # batch x len_query x 1 x hidden_dim
    attention_1 = tf.expand_dims(attention_1, 2)
    # batch x 1 x len_context x hidden_dim
    attention_2 = tf.expand_dims(attention_2, 1)

    # batch x len_query x len_context x hidden_dim
    attention = tf.nn.tanh(attention_1+attention_2)

    # batch x len_query x len_context x 1
    S = tf.einsum("abcd,de->abce", attention, Vc)
    S = tf.squeeze(S, -1) # batch x len_query x len_context

    mask_q = tf.expand_dims(query_mask, 1) # batch x 1 x query_len
    mask_c = tf.expand_dims(context_mask, 1) # batch x 1 x context_len

    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
    c2q = tf.matmul(S_, context) 

    S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
    q2c = tf.matmul(S_T, query)

    return c2q, q2c

def dot_attention(query, context,
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):

    hidden_dim = query.get_shape()[-1]
    Wd = tf.get_variable("Wd", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Vd = tf.get_variable("Vd", dtype=tf.float32,
                                    shape=[hidden_dim, 1],
                                    initializer=initializer)

    # batch x len_query x 1 x hidden_dim
    query_ = tf.expand_dims(query, 2)
    # batch x 1 x len_context x hidden_dim
    context_ = tf.expand_dims(context, 1)

    # batch x len_query x len_context x hidden_dim
    dot_attention = query_ * context_
    dot_attention = tf.einsum("abcd,de->abce", dot_attention, Wd)
    dot_attention = tf.einsum("abce,ef->abcf", dot_attention, Vd)

    # batch x len_query x len_context
    S = tf.squeeze(dot_attention, -1)
    mask_q = tf.expand_dims(query_mask, 1) # batch x 1 x query_len
    mask_c = tf.expand_dims(context_mask, 1) # batch x 1 x context_len

    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
    c2q = tf.matmul(S_, context) 

    S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
    q2c = tf.matmul(S_T, query)

    return c2q, q2c

def minus_attention(query, context, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    
    hidden_dim = query.get_shape()[-1]
    Wm = tf.get_variable("Wm", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Vm = tf.get_variable("Vm", dtype=tf.float32,
                                    shape=[hidden_dim, 1],
                                    initializer=initializer)

    # batch x len_query x 1 x hidden_dim
    query_ = tf.expand_dims(query, 2)
    # batch x 1 x len_context x hidden_dim
    context_ = tf.expand_dims(context, 1)

    # batch x len_query x len_context x hidden_dim
    minus_attention = tf.abs(query_ - context_)

    minus_attention = tf.einsum("abcd,de->abce", minus_attention, Wm)
    minus_attention = tf.einsum("abce,ef->abcf", minus_attention, Vm)

    # batch x len_query x len_context
    S = tf.squeeze(minus_attention, -1)
    mask_q = tf.expand_dims(query_mask, 1) # batch x 1 x query_len
    mask_c = tf.expand_dims(context_mask, 1) # batch x 1 x context_len

    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
    c2q = tf.matmul(S_, context) 

    S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
    q2c = tf.matmul(S_T, query)

    return c2q, q2c

def self_attention(query, context,
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):

    hidden_dim = query.get_shape()[-1]
    Wq_1 = tf.get_variable("Wq_1", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Vq = tf.get_variable("Vq", dtype=tf.float32,
                                    shape=[hidden_dim, 1],
                                    initializer=initializer)

    Wp_1 = tf.get_variable("Wp_1", dtype=tf.float32,
                                    shape=[hidden_dim, hidden_dim],
                                    initializer=initializer)

    Wp_2 = tf.get_variable("Wp_2", dtype=tf.float32,
                                    shape=[hidden_dim, 1],
                                    initializer=initializer)

    # S = tf.matmul(tf.nn.tanh(tf.matmul(query, Wq_1)), Vq_1)
    S = tf.nn.tanh(tf.einsum("abc,cd->abd", query, Wq_1))
    S = tf.einsum("abd,de->abe", S, Vq)

    S = tf.squeeze(S, -1) # batch x query_len

    mask_q = query_mask # batch x query_len

    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_q))
    S_ = tf.expand_dims(S_, axis=-1) # batch x len x 1
    query_attn = tf.reduce_sum(S_ * query, axis=1, keepdims=True) # batch x 1 x hidden_dim

    # batch x context_len x 1
    S = tf.nn.tanh(tf.einsum("abc,cd->abd", context, Wp_1))
    S += tf.nn.tanh(tf.einsum("abc,cd->abd", query_attn, Wp_1))
    S = tf.nn.tanh(S)
    S = tf.einsum("abd,de->abe", S, Vq)

    S = tf.squeeze(S, -1) # batch x context_len

    mask_c = context_mask # batch x context_len
    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
    S_ = tf.expand_dims(S_, axis=-1) # batch x context_len x 1
    context_attn = tf.reduce_sum(S_ * context, axis=1, keepdims=True) # batch x 1 x hidden_dim
    context_attn = tf.squeeze(context_attn, axis=1)
    
    return context_attn
