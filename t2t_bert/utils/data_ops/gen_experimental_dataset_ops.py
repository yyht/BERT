import collections as _collections
import six as _six

from tensorflow.python.framework import tensor_shape as _tensor_shape
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import ops as _ops
from tensorflow.python.eager import execute as _execute

def experimental_directed_interleave_dataset(selector_input_dataset, data_input_datasets, output_types, output_shapes, name=None):
  r"""A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

  Args:
    selector_input_dataset: A `Tensor` of type `variant`.
      A dataset of scalar `DT_INT64` elements that determines which of the
      `N` data inputs should produce the next output element.
    data_input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
      `N` datasets with the same type that will be interleaved according to
      the values of `selector_input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  # Add nodes to the TensorFlow graph.
  if not isinstance(data_input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'data_input_datasets' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % data_input_datasets)
  _attr_N = len(data_input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_directed_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op = _op_def_lib._apply_op_helper(
        "ExperimentalDirectedInterleaveDataset", selector_input_dataset=selector_input_dataset,
                                                 data_input_datasets=data_input_datasets,
                                                 output_types=output_types,
                                                 output_shapes=output_shapes,
                                                 name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result

def experimental_group_by_reducer_dataset(input_dataset, key_func_other_arguments, init_func_other_arguments, reduce_func_other_arguments, finalize_func_other_arguments, key_func, init_func, reduce_func, finalize_func, output_types, output_shapes, name=None):
  r"""Creates a dataset that computes a group-by on `input_dataset`.

  Creates a dataset that computes a group-by on `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    key_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `key_func`.
    init_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `init_func`.
    reduce_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `reduce_func`.
    finalize_func_other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `finalize_func`.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    init_func: A function decorated with @Defun.
      A function mapping a key of type DT_INT64, concatenated with
      `init_func_other_arguments` to the initial reducer state.
    reduce_func: A function decorated with @Defun.
      A function mapping the current reducer state and an element of `input_dataset`,
      concatenated with `reduce_func_other_arguments` to a new reducer state.
    finalize_func: A function decorated with @Defun.
      A function mapping the final reducer state to an output element.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_reducer_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op = _op_def_lib._apply_op_helper(
        "ExperimentalGroupByReducerDataset", input_dataset=input_dataset,
                                             key_func_other_arguments=key_func_other_arguments,
                                             init_func_other_arguments=init_func_other_arguments,
                                             reduce_func_other_arguments=reduce_func_other_arguments,
                                             finalize_func_other_arguments=finalize_func_other_arguments,
                                             key_func=key_func,
                                             init_func=init_func,
                                             reduce_func=reduce_func,
                                             finalize_func=finalize_func,
                                             output_types=output_types,
                                             output_shapes=output_shapes,
                                             name=name)
  _result = _op.outputs[:]
  _result, = _result
  return _result

def experimental_group_by_window_dataset(input_dataset, key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, name=None):
  r"""Creates a dataset that computes a windowed group-by on `input_dataset`.

  // TODO(mrry): Support non-int64 keys.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    key_func_other_arguments: A list of `Tensor` objects.
    reduce_func_other_arguments: A list of `Tensor` objects.
    window_size_func_other_arguments: A list of `Tensor` objects.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    reduce_func: A function decorated with @Defun.
    window_size_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'experimental_group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op = _op_def_lib._apply_op_helper(
        "ExperimentalGroupByWindowDataset", input_dataset=input_dataset,
                                            key_func_other_arguments=key_func_other_arguments,
                                            reduce_func_other_arguments=reduce_func_other_arguments,
                                            window_size_func_other_arguments=window_size_func_other_arguments,
                                            key_func=key_func,
                                            reduce_func=reduce_func,
                                            window_size_func=window_size_func,
                                            output_types=output_types,
                                            output_shapes=output_shapes,
                                            name=name)
  _result = _op.outputs[:]
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
#   name: "ExperimentalAssertNextDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "transformations"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalBytesProducedStatsDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalCSVDataset"
#   input_arg {
#     name: "filenames"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "compression_type"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "header"
#     type: DT_BOOL
#   }
#   input_arg {
#     name: "field_delim"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "use_quote_delim"
#     type: DT_BOOL
#   }
#   input_arg {
#     name: "na_value"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "select_cols"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "record_defaults"
#     type_list_attr: "output_types"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_STRING
#       }
#     }
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalDatasetCardinality"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "cardinality"
#     type: DT_INT64
#   }
# }
# op {
#   name: "ExperimentalDatasetToTFRecord"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "filename"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "compression_type"
#     type: DT_STRING
#   }
# }
# op {
#   name: "ExperimentalDenseToSparseBatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "row_shape"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalDirectedInterleaveDataset"
#   input_arg {
#     name: "selector_input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "data_input_datasets"
#     type: DT_VARIANT
#     number_attr: "N"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalGroupByReducerDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "key_func_other_arguments"
#     type_list_attr: "Tkey_func_other_arguments"
#   }
#   input_arg {
#     name: "init_func_other_arguments"
#     type_list_attr: "Tinit_func_other_arguments"
#   }
#   input_arg {
#     name: "reduce_func_other_arguments"
#     type_list_attr: "Treduce_func_other_arguments"
#   }
#   input_arg {
#     name: "finalize_func_other_arguments"
#     type_list_attr: "Tfinalize_func_other_arguments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "key_func"
#     type: "func"
#   }
#   attr {
#     name: "init_func"
#     type: "func"
#   }
#   attr {
#     name: "reduce_func"
#     type: "func"
#   }
#   attr {
#     name: "finalize_func"
#     type: "func"
#   }
#   attr {
#     name: "Tkey_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Tinit_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Treduce_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Tfinalize_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalGroupByWindowDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "key_func_other_arguments"
#     type_list_attr: "Tkey_func_other_arguments"
#   }
#   input_arg {
#     name: "reduce_func_other_arguments"
#     type_list_attr: "Treduce_func_other_arguments"
#   }
#   input_arg {
#     name: "window_size_func_other_arguments"
#     type_list_attr: "Twindow_size_func_other_arguments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "key_func"
#     type: "func"
#   }
#   attr {
#     name: "reduce_func"
#     type: "func"
#   }
#   attr {
#     name: "window_size_func"
#     type: "func"
#   }
#   attr {
#     name: "Tkey_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Treduce_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Twindow_size_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalIdentityIndexedDataset"
#   input_arg {
#     name: "size"
#     type: DT_UINT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalIgnoreErrorsDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalIndexedDatasetGet"
#   input_arg {
#     name: "materialized"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "index"
#     type: DT_UINT64
#   }
#   output_arg {
#     name: "components"
#     type_list_attr: "output_types"
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalIndexedDatasetMaterialize"
#   input_arg {
#     name: "dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "materialized"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalIteratorGetDevice"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "device"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalLMDBDataset"
#   input_arg {
#     name: "filenames"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalLatencyStatsDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalMapAndBatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "num_parallel_calls"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "drop_remainder"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "preserve_cardinality"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ExperimentalMapDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "use_inter_op_parallelism"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "preserve_cardinality"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ExperimentalMatchingFilesDataset"
#   input_arg {
#     name: "patterns"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalMaterializedIndexDatasetHandle"
#   output_arg {
#     name: "handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "container"
#     type: "string"
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalMaxIntraOpParallelismDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "max_intra_op_parallelism"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalNonSerializableDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalNumaMapAndBatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "num_parallel_calls"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "drop_remainder"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "preserve_cardinality"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ExperimentalParallelInterleaveDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "cycle_length"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "block_length"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "sloppy"
#     type: DT_BOOL
#   }
#   input_arg {
#     name: "buffer_output_elements"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "prefetch_input_elements"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalParseExampleDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "num_parallel_calls"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "dense_defaults"
#     type_list_attr: "Tdense"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "sparse_keys"
#     type: "list(string)"
#     has_minimum: true
#   }
#   attr {
#     name: "dense_keys"
#     type: "list(string)"
#     has_minimum: true
#   }
#   attr {
#     name: "sparse_types"
#     type: "list(type)"
#     has_minimum: true
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_INT64
#         type: DT_STRING
#       }
#     }
#   }
#   attr {
#     name: "Tdense"
#     type: "list(type)"
#     has_minimum: true
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_INT64
#         type: DT_STRING
#       }
#     }
#   }
#   attr {
#     name: "dense_shapes"
#     type: "list(shape)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "sloppy"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ExperimentalPrivateThreadPoolDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "num_threads"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalRandomDataset"
#   input_arg {
#     name: "seed"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "seed2"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalScanDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "initial_state"
#     type_list_attr: "Tstate"
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Tstate"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "preserve_cardinality"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ExperimentalSetStatsAggregatorDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "stats_aggregator"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "counter_prefix"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalSleepDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "sleep_microseconds"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalSlidingWindowDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "window_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "window_shift"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "window_stride"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalSqlDataset"
#   input_arg {
#     name: "driver_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "data_source_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "query"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalStatsAggregatorHandle"
#   output_arg {
#     name: "handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalStatsAggregatorSummary"
#   input_arg {
#     name: "iterator"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "summary"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalThreadPoolDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "thread_pool"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalThreadPoolHandle"
#   output_arg {
#     name: "handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "num_threads"
#     type: "int"
#   }
#   attr {
#     name: "max_intra_op_parallelism"
#     type: "int"
#     default_value {
#       i: 1
#     }
#   }
#   attr {
#     name: "display_name"
#     type: "string"
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "ExperimentalUnbatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ExperimentalUniqueDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\225\001\n\035ExperimentalAssertNextDataset\022\021\n\rinput_dataset\030\025\022\023\n\017transformations\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\221\001\n%ExperimentalBytesProducedStatsDataset\022\021\n\rinput_dataset\030\025\022\007\n\003tag\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\234\002\n\026ExperimentalCSVDataset\022\r\n\tfilenames\030\007\022\024\n\020compression_type\030\007\022\017\n\013buffer_size\030\t\022\n\n\006header\030\n\022\017\n\013field_delim\030\007\022\023\n\017use_quote_delim\030\n\022\014\n\010na_value\030\007\022\017\n\013select_cols\030\t\022\037\n\017record_defaults2\014output_types\032\n\n\006handle\030\025\")\n\014output_types\022\nlist(type)(\0010\001:\t\n\0072\005\001\002\003\t\007\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nD\n\036ExperimentalDatasetCardinality\022\021\n\rinput_dataset\030\025\032\017\n\013cardinality\030\t\nV\n\035ExperimentalDatasetToTFRecord\022\021\n\rinput_dataset\030\025\022\014\n\010filename\030\007\022\024\n\020compression_type\030\007\n\247\001\n%ExperimentalDenseToSparseBatchDataset\022\021\n\rinput_dataset\030\025\022\016\n\nbatch_size\030\t\022\r\n\trow_shape\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\273\001\n%ExperimentalDirectedInterleaveDataset\022\032\n\026selector_input_dataset\030\025\022\032\n\023data_input_datasets\030\025*\001N\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"\014\n\001N\022\003int(\0010\001\n\373\004\n!ExperimentalGroupByReducerDataset\022\021\n\rinput_dataset\030\025\0225\n\030key_func_other_arguments2\031Tkey_func_other_arguments\0227\n\031init_func_other_arguments2\032Tinit_func_other_arguments\022;\n\033reduce_func_other_arguments2\034Treduce_func_other_arguments\022?\n\035finalize_func_other_arguments2\036Tfinalize_func_other_arguments\032\n\n\006handle\030\025\"\020\n\010key_func\022\004func\"\021\n\tinit_func\022\004func\"\023\n\013reduce_func\022\004func\"\025\n\rfinalize_func\022\004func\")\n\031Tkey_func_other_arguments\022\nlist(type)(\001\"*\n\032Tinit_func_other_arguments\022\nlist(type)(\001\",\n\034Treduce_func_other_arguments\022\nlist(type)(\001\".\n\036Tfinalize_func_other_arguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\213\004\n ExperimentalGroupByWindowDataset\022\021\n\rinput_dataset\030\025\0225\n\030key_func_other_arguments2\031Tkey_func_other_arguments\022;\n\033reduce_func_other_arguments2\034Treduce_func_other_arguments\022E\n window_size_func_other_arguments2!Twindow_size_func_other_arguments\032\n\n\006handle\030\025\"\020\n\010key_func\022\004func\"\023\n\013reduce_func\022\004func\"\030\n\020window_size_func\022\004func\")\n\031Tkey_func_other_arguments\022\nlist(type)(\001\",\n\034Treduce_func_other_arguments\022\nlist(type)(\001\"1\n!Twindow_size_func_other_arguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n=\n\"ExperimentalIdentityIndexedDataset\022\010\n\004size\030\027\032\n\n\006handle\030\025\210\001\001\n\202\001\n\037ExperimentalIgnoreErrorsDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\235\001\n\035ExperimentalIndexedDatasetGet\022\020\n\014materialized\030\024\022\t\n\005index\030\027\032\032\n\ncomponents2\014output_types\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nI\n%ExperimentalIndexedDatasetMaterialize\022\013\n\007dataset\030\025\022\020\n\014materialized\030\024\210\001\001\n<\n\035ExperimentalIteratorGetDevice\022\014\n\010resource\030\024\032\n\n\006device\030\007\210\001\001\ny\n\027ExperimentalLMDBDataset\022\r\n\tfilenames\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\213\001\n\037ExperimentalLatencyStatsDataset\022\021\n\rinput_dataset\030\025\022\007\n\003tag\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\245\002\n\036ExperimentalMapAndBatchDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\016\n\nbatch_size\030\t\022\026\n\022num_parallel_calls\030\t\022\022\n\016drop_remainder\030\n\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\" \n\024preserve_cardinality\022\004bool\032\002(\000\n\207\002\n\026ExperimentalMapDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"$\n\030use_inter_op_parallelism\022\004bool\032\002(\001\" \n\024preserve_cardinality\022\004bool\032\002(\000\n?\n ExperimentalMatchingFilesDataset\022\014\n\010patterns\030\007\032\n\n\006handle\030\025\210\001\001\n\251\001\n*ExperimentalMaterializedIndexDatasetHandle\032\n\n\006handle\030\024\"\023\n\tcontainer\022\006string\"\025\n\013shared_name\022\006string\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\251\001\n(ExperimentalMaxIntraOpParallelismDataset\022\021\n\rinput_dataset\030\025\022\034\n\030max_intra_op_parallelism\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\205\001\n\"ExperimentalNonSerializableDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\251\002\n\"ExperimentalNumaMapAndBatchDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\016\n\nbatch_size\030\t\022\026\n\022num_parallel_calls\030\t\022\022\n\016drop_remainder\030\n\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\" \n\024preserve_cardinality\022\004bool\032\002(\000\n\267\002\n%ExperimentalParallelInterleaveDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\020\n\014cycle_length\030\t\022\020\n\014block_length\030\t\022\n\n\006sloppy\030\n\022\032\n\026buffer_output_elements\030\t\022\033\n\027prefetch_input_elements\030\t\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\354\002\n\037ExperimentalParseExampleDataset\022\021\n\rinput_dataset\030\025\022\026\n\022num_parallel_calls\030\t\022\030\n\016dense_defaults2\006Tdense\032\n\n\006handle\030\025\"\035\n\013sparse_keys\022\014list(string)(\001\"\034\n\ndense_keys\022\014list(string)(\001\"%\n\014sparse_types\022\nlist(type)(\001:\007\n\0052\003\001\t\007\"\037\n\006Tdense\022\nlist(type)(\001:\007\n\0052\003\001\t\007\"\035\n\014dense_shapes\022\013list(shape)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"\022\n\006sloppy\022\004bool\032\002(\000\n\230\001\n$ExperimentalPrivateThreadPoolDataset\022\021\n\rinput_dataset\030\025\022\017\n\013num_threads\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\201\001\n\031ExperimentalRandomDataset\022\010\n\004seed\030\t\022\t\n\005seed2\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\225\002\n\027ExperimentalScanDataset\022\021\n\rinput_dataset\030\025\022\027\n\rinitial_state2\006Tstate\022\035\n\017other_arguments2\nTarguments\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\030\n\006Tstate\022\nlist(type)(\0010\001\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\" \n\024preserve_cardinality\022\004bool\032\002(\000\n\276\001\n%ExperimentalSetStatsAggregatorDataset\022\021\n\rinput_dataset\030\025\022\024\n\020stats_aggregator\030\024\022\007\n\003tag\030\007\022\022\n\016counter_prefix\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\223\001\n\030ExperimentalSleepDataset\022\021\n\rinput_dataset\030\025\022\026\n\022sleep_microseconds\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\271\001\n ExperimentalSlidingWindowDataset\022\021\n\rinput_dataset\030\025\022\017\n\013window_size\030\t\022\020\n\014window_shift\030\t\022\021\n\rwindow_stride\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\233\001\n\026ExperimentalSqlDataset\022\017\n\013driver_name\030\007\022\024\n\020data_source_name\030\007\022\t\n\005query\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nf\n!ExperimentalStatsAggregatorHandle\032\n\n\006handle\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\nB\n\"ExperimentalStatsAggregatorSummary\022\014\n\010iterator\030\024\032\013\n\007summary\030\007\210\001\001\n\224\001\n\035ExperimentalThreadPoolDataset\022\021\n\rinput_dataset\030\025\022\017\n\013thread_pool\030\024\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\262\001\n\034ExperimentalThreadPoolHandle\032\n\n\006handle\030\024\"\022\n\013num_threads\022\003int\"#\n\030max_intra_op_parallelism\022\003int\032\002\030\001\"\026\n\014display_name\022\006string\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n}\n\032ExperimentalUnbatchDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n|\n\031ExperimentalUniqueDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001")



