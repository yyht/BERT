import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

def query_context_attention(query, context, max_query_len, max_context_len, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):
        context_ = tf.transpose(context, [0,2,1])
        hiddem_dim = query.get_shape()[-1]

        attn_W = tf.get_variable("AttnW", dtype=tf.float32,
                                    shape=[hiddem_dim, hiddem_dim],
                                    initializer=initializer)

        weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]])

        S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

        mask_q = tf.expand_dims(query_mask, 1)
        mask_c = tf.expand_dims(context_mask, 1)

        S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
        c2q = tf.matmul(S_, context)

        S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        q2c = tf.matmul(S_T, query)

        query_attention_outputs = tf.concat([query, c2q, query-c2q, query*c2q], axis=-1)
        query_attention_outputs *= tf.expand_dims(tf.cast(query_mask, tf.float32), -1)

        context_attention_outputs = tf.concat([context, q2c, context-q2c, context*q2c], axis=-1)
        context_attention_outputs *= tf.expand_dims(tf.cast(context_mask, tf.float32), -1)

        return query_attention_outputs, context_attention_outputs

def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant