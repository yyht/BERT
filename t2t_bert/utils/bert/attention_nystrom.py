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

from utils.bert import bert_utils
from utils.bert import layer_norm_utils
from utils.bert import dropout_utils
from utils.conv_utils import dynamic_conv_kernel

def iterative_inv(mat, n_iter=6):

  """
  https://downloads.hindawi.com/journals/aaa/2014/563787.pdf
  A New Iterative Method for Finding Approximate Inverses of
  Complex Matrices
  """

  mat_shape = bert_utils.get_shape_list(mat, expected_rank=[2,3,4])
  I = tf.cast(tf.eye(mat_shape[-1]), dtype=tf.float32)
  K = tf.identity(mat) 
  # [B, N, n-landmarks, n-landmarks]
  V = 1 / (tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-2)) * tf.reduce_max(tf.reduce_sum(tf.abs(K), axis=-1))) * tf.transpose(K, [0,1,3,2])

  for _ in range(n_iter):
      KV = tf.matmul(K, V)
      V = tf.matmul(0.25 * V, 13 * I - tf.matmul(KV, 15 * I - tf.matmul(KV, 7 * I - KV)))
  # [B, N, n-landmarks, n-landmarks]
  V = tf.stop_gradient(V)
  return V

def attention_nystrom_layer(from_tensor,
              to_tensor,
              attention_mask=None,
              num_attention_heads=1,
              size_per_head=512,
              query_act=None,
              key_act=None,
              value_act=None,
              attention_probs_dropout_prob=0.0,
              initializer_range=0.02,
              do_return_2d_tensor=False,
              batch_size=None,
              from_seq_length=None,
              to_seq_length=None,
              attention_fixed_size=None,
              dropout_name=None,
              structural_attentions="none",
              is_training=False,
              num_landmarks=64,
              use_conv=False,
              original_mask=None,
              **kargs):

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = bert_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = bert_utils.get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  if attention_fixed_size:
    attention_head_size = attention_fixed_size
    tf.logging.info("==apply attention_fixed_size==")
  else:
    attention_head_size = size_per_head
    tf.logging.info("==apply attention_original_size==")

  from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
  to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * attention_head_size,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * attention_head_size,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * attention_head_size,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                   num_attention_heads, 
                   from_seq_length,
                   attention_head_size)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, 
                  num_attention_heads,
                  to_seq_length, 
                  attention_head_size)

  if num_landmarks == from_seq_length:
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                  1.0 / math.sqrt(float(attention_head_size)))


    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder
    
    attention_probs = tf.exp(tf.nn.log_softmax(attention_scores))
    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, attention_head_size])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

  else:

    # [B, N, T, H]
    value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, attention_head_size])

    # [batch_size, N, N-landmarks, from_seq_length//N-landmarks, H]
    query_layer_landmarks = tf.reshape(
        query_layer, [batch_size, num_attention_heads, num_landmarks,
                        from_seq_length//num_landmarks, width])

    # [batch_size, N, N-landmarks, to_seq_length//N-landmarks, H]
    key_layer_landmarks = tf.reshape(
        query_layer, [batch_size, num_attention_heads, num_landmarks,
                        to_seq_length//num_landmarks, width])

    # [batch_size, N, N-landmarks, H]
    query_layer_landmarks = tf.reduce_mean(query_layer_landmarks, axis=-2)
    
    # [batch_size, N, N-landmarks, H]
    key_layer_landmarks = tf.reduce_mean(key_layer_landmarks, axis=-2)

    # query-layer: [B, N, F, H]
    # key_layer_landmarks: [B, N, n-landmarks, H]
    # kernel_1: [B, N, F, n-landmarks]
    kernel_1 = tf.nn.softmax(tf.matmul(query_layer, tf.transpose(key_layer_landmarks, [0, 1, 3, 2])), axis=-1)
    
    # query_layer_landmarks: [B, N, n-landmarks, H]
    # key_layer_landmarks:   [B, N, n-landmarks, H]
    # kernel_2: [B, N, n-landmarks, n-landmarks]
    kernel_2 = tf.nn.softmax(tf.matmul(query_layer_landmarks, tf.transpose(key_layer_landmarks, [0, 1, 3, 2])), axis=-1)

    # query_layer_landmarks: [B, N, n-landmarks, H]
    # key_layer: [B, N, T, H]
    # kernel_3: [B, N, n-landmarks, T]
    kernel_3 = tf.matmul(query_layer_landmarks, tf.transpose(key_layer, [0, 1, 3 ,2]))
    if original_mask is not None:
      # `attention_mask` = [B, 1, T]
      original_mask = tf.expand_dims(original_mask, axis=[1])
      # `attention_mask` = [B, 1, 1, T]
      original_mask = tf.expand_dims(original_mask, axis=[2])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(original_mask, tf.float32)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      kernel_3 += adder

    # kernel_3: [B, N, n-landmarks, T]
    kernel_3 = tf.nn.softmax(kernel_3, axis=-1)

    # kernel_1: [B, N, F, n-landmarks]
    # iterative_inv(kernel_2) : [B, N, n-landmarks, n-landmarks]
    # kernel_12:[B, N, F, n-landmarks]
    kernel_12 = tf.matmul(kernel_1, iterative_inv(kernel_2))

    # kernel_3: [B, N, n-landmarks, T]
    # value_layer: [B, N, T, H]
    # kernel_3_value: [B, N, n-landmarks, H]
    kernel_3_value = tf.matmul(kernel_3, value_layer)
    
    # kernel_12:[B, N, F, n-landmarks]
    # kernel_3_value: [B, N, n-landmarks, H]
    # context_layer: [B, N, F, H]
    context_layer = tf.matmul(kernel_12, kernel_3_value)

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*V]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * attention_head_size])
  else:
    # `context_layer` = [B, F, N*V]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * attention_head_size])

  return context_layer, kernel_3, value_layer
