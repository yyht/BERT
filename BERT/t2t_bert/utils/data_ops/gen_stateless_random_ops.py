"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: stateless_random_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch


def stateless_multinomial(logits, num_samples, seed, output_dtype=_dtypes.int64, name=None):
  r"""Draws samples from a multinomial distribution.

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
      represents the unnormalized log probabilities for all classes.
    num_samples: A `Tensor` of type `int32`.
      0-D.  Number of independent samples to draw for each row slice.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_dtype`.
  """
  # Add nodes to the TensorFlow graph.
  if output_dtype is None:
    output_dtype = _dtypes.int64
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessMultinomial", logits=logits, num_samples=num_samples,
                                seed=seed, output_dtype=output_dtype,
                                name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result

def stateless_random_normal(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessRandomNormal", shape=shape, seed=seed, dtype=dtype,
                                 name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result

def stateless_random_uniform(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessRandomUniform", shape=shape, seed=seed, dtype=dtype,
                                  name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result


def stateless_random_uniform_int(shape, seed, minval, maxval, name=None):
  r"""Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values follow a uniform distribution in the range `[minval, maxval)`.

  The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  # Add nodes to the TensorFlow graph.
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessRandomUniformInt", shape=shape, seed=seed, minval=minval,
                                     maxval=maxval, name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result


def stateless_truncated_normal(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessTruncatedNormal", shape=shape, seed=seed, dtype=dtype,
                                    name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"), "Tseed",
            _op.get_attr("Tseed"))
  _execute.record_gradient(
      "StatelessTruncatedNormal", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "StatelessMultinomial"
#   input_arg {
#     name: "logits"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "num_samples"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "seed"
#     type_attr: "Tseed"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "output_dtype"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_INT64
#         type: DT_BFLOAT16
#         type: DT_UINT16
#         type: DT_HALF
#         type: DT_UINT32
#         type: DT_UINT64
#       }
#     }
#   }
#   attr {
#     name: "Tseed"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "output_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessRandomNormal"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type_attr: "Tseed"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_BFLOAT16
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "Tseed"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessRandomUniform"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type_attr: "Tseed"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_BFLOAT16
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "Tseed"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessRandomUniformInt"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type_attr: "Tseed"
#   }
#   input_arg {
#     name: "minval"
#     type_attr: "dtype"
#   }
#   input_arg {
#     name: "maxval"
#     type_attr: "dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "Tseed"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessTruncatedNormal"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type_attr: "Tseed"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_BFLOAT16
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   attr {
#     name: "Tseed"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\265\001\n\024StatelessMultinomial\022\013\n\006logits\"\001T\022\017\n\013num_samples\030\003\022\r\n\004seed\"\005Tseed\032\026\n\006output\"\014output_dtype\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\"\031\n\005Tseed\022\004type\032\0020\t:\006\n\0042\002\003\t\" \n\014output_dtype\022\004type\032\0020\t:\006\n\0042\002\003\t\n\222\001\n\025StatelessRandomNormal\022\n\n\005shape\"\001T\022\r\n\004seed\"\005Tseed\032\017\n\006output\"\005dtype\"\033\n\005dtype\022\004type\032\0020\001:\010\n\0062\004\023\016\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\031\n\005Tseed\022\004type\032\0020\t:\006\n\0042\002\003\t\n\223\001\n\026StatelessRandomUniform\022\n\n\005shape\"\001T\022\r\n\004seed\"\005Tseed\032\017\n\006output\"\005dtype\"\033\n\005dtype\022\004type\032\0020\001:\010\n\0062\004\023\016\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\031\n\005Tseed\022\004type\032\0020\t:\006\n\0042\002\003\t\n\256\001\n\031StatelessRandomUniformInt\022\n\n\005shape\"\001T\022\r\n\004seed\"\005Tseed\022\017\n\006minval\"\005dtype\022\017\n\006maxval\"\005dtype\032\017\n\006output\"\005dtype\"\025\n\005dtype\022\004type:\006\n\0042\002\003\t\"\021\n\001T\022\004type:\006\n\0042\002\003\t\"\031\n\005Tseed\022\004type\032\0020\t:\006\n\0042\002\003\t\n\225\001\n\030StatelessTruncatedNormal\022\n\n\005shape\"\001T\022\r\n\004seed\"\005Tseed\032\017\n\006output\"\005dtype\"\033\n\005dtype\022\004type\032\0020\001:\010\n\0062\004\023\016\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\"\031\n\005Tseed\022\004type\032\0020\t:\006\n\0042\002\003\t")
