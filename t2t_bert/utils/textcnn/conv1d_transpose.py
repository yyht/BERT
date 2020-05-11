
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers
import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  if value is None:
    value = [1]
  elif not isinstance(value, list):
    value = [value]

  current_n = len(value)
  if current_n == n + 2:
    return value
  elif current_n == 1:
    value = list((value[0],) * n)
  elif current_n == n:
    value = list(value)
  else:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, current_n))

  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]

def conv1d_transpose(
    input,  # pylint: disable=redefined-builtin
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NWC",
    dilations=None,
    name=None):
  """The transpose of `conv1d`.
  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is actually the transpose (gradient) of `conv1d`
  rather than an actual deconvolution.
  Args:
    input: A 3-D `Tensor` of type `float` and shape
      `[batch, in_width, in_channels]` for `NWC` data format or
      `[batch, in_channels, in_width]` for `NCW` data format.
    filters: A 3-D `Tensor` with the same type as `value` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, if
      `output_shape` is not at 3-element vector, if `padding` is other than
      `'VALID'` or `'SAME'`, or if `data_format` is invalid.
  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv1d_transpose",
                      [input, filters, output_shape]) as name:
    # The format could be either NWC or NCW, map to NHWC or NCHW
    if data_format is None or data_format == "NWC":
      data_format = "NHWC"
      spatial_start_dim = 1
      channel_index = 2
    elif data_format == "NCW":
      data_format = "NCHW"
      spatial_start_dim = 2
      channel_index = 1
    else:
      raise ValueError("data_format must be \"NWC\" or \"NCW\".")

    # Reshape the input tensor to [batch, 1, in_width, in_channels]
    strides = [1] + _get_sequence(strides, 1, channel_index, "stride")
    dilations = [1] + _get_sequence(dilations, 1, channel_index, "dilations")

    input = array_ops.expand_dims(input, spatial_start_dim)
    filters = array_ops.expand_dims(filters, 0)
    output_shape = list(output_shape) if not isinstance(
        output_shape, ops.Tensor) else output_shape
    output_shape = array_ops.concat([output_shape[: spatial_start_dim], [1],
                                     output_shape[spatial_start_dim:]], 0)

    result = gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
    return array_ops.squeeze(result, spatial_start_dim)