from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re

import numpy as np
import six
import tensorflow as tf
from utils.bert import bert_utils
from utils.bert import dropout_utils

stable_dropout = dropout_utils.ReuseDropout()

def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)

def dropout(input_tensor, dropout_prob, dropout_name=None):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return tf.identity(input_tensor)
  if dropout_name:
    output = stable_dropout.dropout(input_tensor, dropout_prob, dropout_name)
  else:
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
  # return layer_norm_utils.layer_norm(
  #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None, dropout_name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob, dropout_name=dropout_name)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def attention_layer(from_tensor,
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
                    conv_kernel_size=9,
                    head_ratio=2,
                    conv_type=1,
                    from_tensor_mask=None,
                    to_tensor_mask=None):
  """Performs several types of attention
  1) multi-headed attention from `from_tensor` to `to_tensor`.

  By default, this is an implementation of multi-headed attention based on "Attention is all 
  you Need". If `from_tensor` and `to_tensor` are the same, then this is self-attention. 
  Each timestep in `from_tensor` attends to the corresponding sequence in `to_tensor`, 
  and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and `to_tensor` into "key"
  and "value" tensors. These are (effectively) a list of tensors of length `num_attention_heads`, 
  where each tensor is of shape [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are softmaxed to obtain attention 
  probabilities. The value tensors are then interpolated by these probabilities, then concatenated back 
  to a single tensor and returned.

  In practice, the multi-headed attention are done with transposes and reshapes rather than actual 
  separate tensors.

  2) mixed attention with span-based dynamic convolution with `from_tensor` and `to_tensor`.
  By setting conv_type to "sdconv", the layer will perform mixed attetion which is a mixture of 
  self-attention and span-based dynamic convolution.
  
  If conv_type is set to 'sdconv', this function will additionally generate a span-aware "key" tensor
  which will be multiplied to the "query" tensor and generate a span-based dynamic conv kernel. The kernel
  will then convolve the "value" tensor to produce the output which will be concat with the self-attention
  heads' output for further processing.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    conv_kernel_size: (Optional) Convolution kernel size for span-based dynamic 
    convolution.
    head_ratio: (Optional) Ratio gamma to reduce the number of attention heads. 
    conv_tpye: (Optional) Which convolution tpye to use. One of ["noconv", "sdconv"]. 
    By default "noconv" is used and the attention layer only uses self-attention.
  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor
  def reshape_for_conv(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads*width])
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
    if batch_size is None or from_seq_length is None or to_seq_length is None:
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

  from_tensor_2d = bert_utils.reshape_to_matrix(from_tensor)
  to_tensor_2d = bert_utils.reshape_to_matrix(to_tensor)


  new_num_attention_heads = int(num_attention_heads/head_ratio)
  if new_num_attention_heads<1:
    head_ratio=num_attention_heads
    num_attention_heads=1
  else:
    num_attention_heads=new_num_attention_heads

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  if conv_type in ['sdconv']:
    # [B,T, N*H]
    key_conv_attn_layer_input = reshape_for_conv(to_tensor_2d, batch_size, num_attention_heads*head_ratio,
                                    to_seq_length, size_per_head)

    if from_tensor_mask is not None and to_tensor_mask is not None:
      to_tensor_2d_mask = tf.cast(to_tensor_mask, tf.float32)[:, :, None]
      from_tensor_2d_mask = tf.cast(from_tensor_mask, tf.float32)[:, :, None]
      key_conv_attn_layer_input *= to_tensor_2d_mask
      tf.logging.info("== apply conv seq-masking on sequence padding ==")

    key_conv_attn_layer = tf.layers.separable_conv1d(key_conv_attn_layer_input,
        num_attention_heads * size_per_head,
        conv_kernel_size,
        padding='same',
        activation=value_act,
        depthwise_initializer=create_initializer(1/conv_kernel_size),
        pointwise_initializer=create_initializer(initializer_range),
        name="conv_attn_key")

    if from_tensor_mask is not None and to_tensor_mask is not None:
      key_conv_attn_layer *= to_tensor_2d_mask
      tf.logging.info("== apply conv seq-masking on sequence padding ==")

    # [B*T, N*H]
    key_conv_attn_layer = bert_utils.reshape_to_matrix(key_conv_attn_layer)

    # conv_attn_layer = tf.multiply(key_conv_attn_layer, query_layer)

    query_gate = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="conv_query_gate",
        kernel_initializer=create_initializer(initializer_range))

    conv_gated = tf.nn.sigmoid(tf.nn.dropout(query_gate, 1-attention_probs_dropout_prob))
    conv_attn_layer = key_conv_attn_layer * conv_gated + query_layer * (1-conv_gated)

    # [B*T, N*K]
    conv_kernel_layer = tf.layers.dense(
        conv_attn_layer,
        num_attention_heads * conv_kernel_size,
        activation=value_act,
        name="conv_attn_kernel",
        kernel_initializer=create_initializer(initializer_range))
    # [B*T*N,K,1]
    conv_kernel_layer = tf.reshape(conv_kernel_layer, 
      [batch_size*to_seq_length*num_attention_heads, conv_kernel_size, 1])
    
    # conv_kernel_layer = tf.nn.softmax(conv_kernel_layer, axis=1)
    attention_probs = tf.exp(tf.nn.log_softmax(conv_kernel_layer+1e-10))
    
    paddings = tf.constant([[0, 0,], [int((conv_kernel_size-1)/2), int((conv_kernel_size-1)/2)],[0,0]])
    
    conv_out_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="conv_attn_point",
        kernel_initializer=create_initializer(initializer_range))
    # [B,T, N*H]
    conv_out_layer = tf.reshape(conv_out_layer,[batch_size,to_seq_length,num_attention_heads * size_per_head])
    if from_tensor_mask is not None and to_tensor_mask is not None:
      conv_out_layer *= to_tensor_2d_mask
      tf.logging.info("== apply conv seq-masking on sequence padding ==")

    conv_out_layer = tf.pad(conv_out_layer, paddings, "CONSTANT")
    # unfold [B,T, N*H, K]
    unfold_conv_out_layer = tf.stack(
      [tf.slice(conv_out_layer, [0, i, 0],[batch_size,to_seq_length,num_attention_heads * size_per_head]) for i in range(conv_kernel_size)],-1)
    # following only work for gpu version
    # conv_out_layer = tf.reshape(conv_out_layer,[batch_size,to_seq_length,num_attention_heads * size_per_head,1])
    # unfold_conv_out_layer = tf.extract_image_patches(images=conv_out_layer, sizes=[1, conv_kernel_size, 1, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')

    conv_out_layer = tf.reshape(unfold_conv_out_layer,
      [batch_size*to_seq_length*num_attention_heads ,size_per_head, conv_kernel_size])

    conv_out_layer = tf.matmul(conv_out_layer, conv_kernel_layer)

    conv_out_layer = tf.reshape(conv_out_layer, [batch_size*to_seq_length, num_attention_heads*size_per_head])


  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

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

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  # attention_probs = tf.nn.softmax(attention_scores)
  attention_probs = tf.exp(tf.nn.log_softmax(attention_scores+1e-10))

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])


  if conv_type in ["sdconv"]:
    # only applicable for self-attention, will cause error if from_seq_length not equal to_seq_length
    assert from_seq_length==to_seq_length
    conv_out = tf.reshape(
        conv_out_layer,
        [batch_size , from_seq_length, num_attention_heads , size_per_head])
    context_layer = tf.concat([context_layer, conv_out],2)
    num_attention_heads = num_attention_heads*2

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer, attention_probs, value_layer

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      conv_kernel_size=3,
                      head_ratio=2,
                      conv_type="noconv",
                      **kargs):
  """Extension of Multi-headed, multi-layer Transformer from "Attention is All You Need".
  
  Addition args are add for span-based dynamic convolution and ConvBERT.
  
  For more detail of transformer, see the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    linear_groups: Number of groups for grouped linear operator to apply on
      feed-forward module.
    conv_kernel_size: Convolution kernel size for span-based dynamic convolution.
    head_ratio: Ratio gamma to reduce the number of attention heads.
    conv_tpye: Which convolution tpye to use. One of ["noconv", "sdconv"]. By default
      "noconv" is used and the attention layer only uses self-attention.


  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = bert_utils.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = bert_utils.reshape_to_matrix(input_tensor)

  attn_maps = []
  all_layer_outputs = []
  all_value_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head, probs, value_layer = attention_layer(
              from_tensor=prev_output,
              to_tensor=prev_output,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              conv_kernel_size=conv_kernel_size,
              head_ratio=head_ratio,
              conv_type=conv_type,
              from_tensor_mask=kargs.get('from_tensor_mask', None),
              to_tensor_mask=kargs.get('to_tensor_mask', None))
          attention_heads.append(attention_head)
          attn_maps.append(probs)
          all_value_outputs.append(value_layer)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        with tf.variable_scope("output"):
          # Run a linear projection of `hidden_size` then add a residual
          # with `layer_input`.
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          

          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + prev_output)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))
      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        prev_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))

        prev_output = dropout(prev_output, hidden_dropout_prob)
        prev_output = layer_norm(prev_output + attention_output)
        all_layer_outputs.append(prev_output)

  attn_maps = tf.stack(attn_maps, 0)
  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = bert_utils.reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs, attn_maps, all_value_outputs
  else:
    final_output = bert_utils.reshape_from_matrix(prev_output, input_shape)
    return final_output, attn_maps, all_value_outputs