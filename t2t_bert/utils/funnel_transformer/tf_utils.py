import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
# from tensorflow.python.util.compat import collections_abc
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.constant_op import constant

def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
	"""Returns a mask tensor representing the first N positions of each cell.
	If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
	dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with
	```
	mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
	```
	Examples:
	```python
	tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
																	#  [True, True, True, False, False],
																	#  [True, True, False, False, False]]
	tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
																		#   [True, True, True]],
																		#  [[True, True, False],
																		#   [False, False, False]]]
	```
	Args:
		lengths: integer tensor, all its values <= maxlen.
		maxlen: scalar integer tensor, size of last dimension of returned tensor.
			Default is the maximum value in `lengths`.
		dtype: output type of the resulting tensor.
		name: name of the op.
	Returns:
		A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
	Raises:
		ValueError: if `maxlen` is not a scalar.
	"""
	with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
		lengths = ops.convert_to_tensor(lengths)

		if maxlen is None:
			maxlen = gen_math_ops._max(lengths, _all_dimensions(lengths))
			maxlen = gen_math_ops.maximum(constant(0, maxlen.dtype), maxlen)
		else:
			maxlen = ops.convert_to_tensor(maxlen)
		if maxlen.get_shape().ndims is not None and maxlen.get_shape().ndims != 0:
			raise ValueError("maxlen must be scalar for sequence_mask")

		# The basic idea is to compare a range row vector of size maxlen:
		# [0, 1, 2, 3, 4]
		# to length as a matrix with 1 column: [[1], [3], [2]].
		# Because of broadcasting on both arguments this comparison results
		# in a matrix of size (len(lengths), maxlen)
		row_vector = gen_math_ops._range(
				constant(0, maxlen.dtype), maxlen, constant(1, maxlen.dtype))
		# Since maxlen >= max(lengths), it is safe to use maxlen as a cast
		# authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
		matrix = gen_math_ops.cast(expand_dims(lengths, -1), maxlen.dtype)
		result = row_vector < matrix

		if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
			return result
		else:
			return gen_math_ops.cast(result, dtype)

def rank(input, name=None):
	# pylint: disable=redefined-builtin
	"""Returns the rank of a tensor.
	Returns a 0-D `int32` `Tensor` representing the rank of `input`.
	For example:
	```python
	# shape of tensor 't' is [2, 2, 3]
	t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
	tf.rank(t)  # 3
	```
	**Note**: The rank of a tensor is not the same as the rank of a matrix. The
	rank of a tensor is the number of indices required to uniquely select each
	element of the tensor. Rank is also known as "order", "degree", or "ndims."
	Args:
		input: A `Tensor` or `SparseTensor`.
		name: A name for the operation (optional).
	Returns:
		A `Tensor` of type `int32`.
	@compatibility(numpy)
	Equivalent to np.ndim
	@end_compatibility
	"""
	return rank_internal(input, name, optimize=True)


def rank_internal(input, name=None, optimize=True):
	# pylint: disable=redefined-builtin
	"""Returns the rank of a tensor.
	Args:
		input: A `Tensor` or `SparseTensor`.
		name: A name for the operation (optional).
		optimize: if true, encode the rank as a constant when possible.
	Returns:
		A `Tensor` of type `int32`.
	"""
	with ops.name_scope(name, "Rank", [input]) as name:
		if isinstance(
				input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
			return gen_array_ops.size(input.dense_shape, name=name)
		else:
			input = ops.convert_to_tensor(input)
			input_shape = input.get_shape()
			if optimize and input_shape.ndims is not None:
				return constant(input_shape.ndims, dtypes.int32, name=name)
			return gen_array_ops.rank(input, name=name)

def _all_dimensions(x):
	"""Returns a 1D-tensor listing all dimensions in x."""
	# Fast path: avoid creating Rank and Range ops if ndims is known.
	if isinstance(x, ops.Tensor) and x.get_shape().ndims is not None:
		return constant_op.constant(
				np.arange(x.get_shape().ndims), dtype=dtypes.int32)
	if (isinstance(x, sparse_tensor.SparseTensor) and
			x.dense_shape.get_shape().is_fully_defined()):
		r = x.dense_shape.get_shape().dims[0].value  # sparse.dense_shape is 1-D.
		return constant_op.constant(np.arange(r), dtype=dtypes.int32)

	# Otherwise, we rely on `range` and `rank` to do the right thing at runtime.
	return gen_math_ops._range(0, rank(x), 1)

def expand_dims_v2(input, axis, name=None):
	"""Returns a tensor with an additional dimension inserted at index `axis`.
	Given a tensor `input`, this operation inserts a dimension of size 1 at the
	dimension index `axis` of `input`'s shape. The dimension index `axis` starts
	at zero; if you specify a negative number for `axis` it is counted backward
	from the end.
	This operation is useful if you want to add a batch dimension to a single
	element. For example, if you have a single image of shape `[height, width,
	channels]`, you can make it a batch of one image with `expand_dims(image, 0)`,
	which will make the shape `[1, height, width, channels]`.
	Examples:
	>>> t = [[1, 2, 3],[4, 5, 6]] # shape [2, 3]
	>>> tf.expand_dims(t, 0)
	<tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
	array([[[1, 2, 3],
					[4, 5, 6]]], dtype=int32)>
	>>> tf.expand_dims(t, 1)
	<tf.Tensor: shape=(2, 1, 3), dtype=int32, numpy=
	array([[[1, 2, 3]],
				 [[4, 5, 6]]], dtype=int32)>
	>>> tf.expand_dims(t, 2)
	<tf.Tensor: shape=(2, 3, 1), dtype=int32, numpy=
	array([[[1],
					[2],
					[3]],
				 [[4],
					[5],
					[6]]], dtype=int32)>
	>>> tf.expand_dims(t, -1) # Last dimension index. In this case, same as 2.
	<tf.Tensor: shape=(2, 3, 1), dtype=int32, numpy=
	array([[[1],
					[2],
					[3]],
				 [[4],
					[5],
					[6]]], dtype=int32)>
	This operation is related to:
	*   `tf.squeeze`, which removes dimensions of size 1.
	*   `tf.reshape`, which provides more flexible reshaping capability
	Args:
		input: A `Tensor`.
		axis: Integer specifying the dimension index at which to expand the
			shape of `input`. Given an input of D dimensions, `axis` must be in range
			`[-(D+1), D]` (inclusive).
		name: Optional string. The name of the output `Tensor`.
	Returns:
		A tensor with the same data as `input`, with an additional dimension
		inserted at the index specified by `axis`.
	Raises:
		ValueError: If `axis` is not specified.
		InvalidArgumentError: If `axis` is out of range `[-(D+1), D]`.
	"""
	return gen_array_ops.expand_dims(input, axis, name)

def _get_sequence(value, n, channel_index, name):
	"""Formats a value input for gen_nn_ops."""
	if value is None:
		value = [1]
	else:
		value = [value]
	# elif not isinstance(value, collections_abc.Sized):
	#   value = [value]

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

def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
	"""Performs the max pooling on the input.
	Note internally this op reshapes and uses the underlying 2d operation.
	Args:
		input: A 3-D `Tensor` of the format specified by `data_format`.
		ksize: An int or list of `ints` that has length `1` or `3`. The size of the
			window for each dimension of the input tensor.
		strides: An int or list of `ints` that has length `1` or `3`. The stride of
			the sliding window for each dimension of the input tensor.
		padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
			the "returns" section of `tf.nn.convolution` for details.
		data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
		name: A name for the operation (optional).
	Returns:
		A `Tensor` of format specified by `data_format`.
		The max pooled output tensor.
	"""
	with ops.name_scope(name, "MaxPool1d", [input]) as name:
		if data_format is None:
			data_format = "NWC"
		channel_index = 1 if data_format.startswith("NC") else 2
		ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
		strides = [1] + _get_sequence(strides, 1, channel_index, "strides")

		expanding_dim = 1 if data_format == "NWC" else 2
		data_format = "NHWC" if data_format == "NWC" else "NCHW"

		input = expand_dims_v2(input, expanding_dim)
		result = gen_nn_ops.max_pool(
				input,
				ksize=ksize,
				strides=strides,
				padding=padding,
				data_format=data_format,
				name=name)
		return array_ops.squeeze(result, expanding_dim)

def avg_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):  # pylint: disable=redefined-builtin
	"""Performs the average pooling on the input.
	Each entry in `output` is the mean of the corresponding size `ksize`
	window in `value`.
	Note internally this op reshapes and uses the underlying 2d operation.
	Args:
		input: A 3-D `Tensor` of the format specified by `data_format`.
		ksize: An int or list of `ints` that has length `1` or `3`. The size of the
			window for each dimension of the input tensor.
		strides: An int or list of `ints` that has length `1` or `3`. The stride of
			the sliding window for each dimension of the input tensor.
		padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
			the "returns" section of `tf.nn.convolution` for details.
		data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
		name: A name for the operation (optional).
	Returns:
		A `Tensor` of format specified by `data_format`.
		The max pooled output tensor.
	"""
	with ops.name_scope(name, "AvgPool1D", [input]) as name:
		if data_format is None:
			data_format = "NWC"
		channel_index = 1 if data_format.startswith("NC") else 2
		ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
		strides = [1] + _get_sequence(strides, 1, channel_index, "strides")

		expanding_dim = 1 if data_format == "NWC" else 2
		data_format = "NHWC" if data_format == "NWC" else "NCHW"

		input = expand_dims_v2(input, expanding_dim)
		result = gen_nn_ops.avg_pool(
				input,
				ksize=ksize,
				strides=strides,
				padding=padding,
				data_format=data_format,
				name=name)
		return array_ops.squeeze(result, expanding_dim)


def get_positive_axis(axis, ndims, axis_name="axis", ndims_name="ndims"):
	"""Validate an `axis` parameter, and normalize it to be positive.
	If `ndims` is known (i.e., not `None`), then check that `axis` is in the
	range `-ndims <= axis < ndims`, and return `axis` (if `axis >= 0`) or
	`axis + ndims` (otherwise).
	If `ndims` is not known, and `axis` is positive, then return it as-is.
	If `ndims` is not known, and `axis` is negative, then report an error.
	Args:
		axis: An integer constant
		ndims: An integer constant, or `None`
		axis_name: The name of `axis` (for error messages).
		ndims_name: The name of `ndims` (for error messages).
	Returns:
		The normalized `axis` value.
	Raises:
		ValueError: If `axis` is out-of-bounds, or if `axis` is negative and
			`ndims is None`.
	"""
	if not isinstance(axis, int):
		raise TypeError("%s must be an int; got %s" %
										(axis_name, type(axis).__name__))
	if ndims is not None:
		if 0 <= axis < ndims:
			return axis
		elif -ndims <= axis < 0:
			return axis + ndims
		else:
			raise ValueError("%s=%s out of bounds: expected %s<=%s<%s" %
											 (axis_name, axis, -ndims, axis_name, ndims))
	elif axis < 0:
		raise ValueError("%s may only be negative if %s is statically known." %
										 (axis_name, ndims_name))
	return axis


# This op is intended to exactly match the semantics of numpy.repeat, with
# one exception: numpy.repeat has special (and somewhat non-intuitive) behavior
# when axis is not specified.  Rather than implement that special behavior, we
# simply make `axis` be a required argument.
#
# External (OSS) `tf.repeat` feature request:
# https://github.com/tensorflow/tensorflow/issues/8246
def repeat_with_axis(data, repeats, axis, name=None):
	"""Repeats elements of `data`.
	Args:
		data: An `N`-dimensional tensor.
		repeats: A 1-D integer tensor specifying how many times each element in
			`axis` should be repeated.  `len(repeats)` must equal `data.shape[axis]`.
			Supports broadcasting from a scalar value.
		axis: `int`.  The axis along which to repeat values.  Must be less than
			`max(N, 1)`.
		name: A name for the operation.
	Returns:
		A tensor with `max(N, 1)` dimensions.  Has the same shape as `data`,
		except that dimension `axis` has size `sum(repeats)`.
	Example usage:
	>>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
	<tf.Tensor: shape=(5,), dtype=string,
	numpy=array([b'a', b'a', b'a', b'c', b'c'], dtype=object)>
	>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
	<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
	array([[1, 2],
				 [1, 2],
				 [3, 4],
				 [3, 4],
				 [3, 4]], dtype=int32)>
	>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
	<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
	array([[1, 1, 2, 2, 2],
				 [3, 3, 4, 4, 4]], dtype=int32)>
	"""
	if not isinstance(axis, int):
		raise TypeError("axis must be an int; got %s" % type(axis).__name__)

	with ops.name_scope(name, "Repeat", [data, repeats]):
		data = ops.convert_to_tensor(data, name="data")
		repeats = tf.cast(repeats, tf.int32)
		# repeats = tile_one_dimension.convert_to_int_tensor(repeats, name="repeats")
		repeats.shape.with_rank_at_most(1)

		# If `data` is a scalar, then upgrade it to a vector.
		data = _with_nonzero_rank(data)
		data_shape = array_ops.shape(data)

		# If `axis` is negative, then convert it to a positive value.
		axis = get_positive_axis(axis, len(data.shape.as_list()), ndims_name="rank(data)")
		# Check data Tensor shapes.
		if repeats.shape.ndims == 1:
			data.shape.dims[axis].assert_is_compatible_with(repeats.shape[0])

		# If we know that `repeats` is a scalar, then we can just tile & reshape.
		if repeats.shape.ndims == 0:
			expanded = array_ops.expand_dims(data, axis + 1)
			tiled = tile_one_dimension(expanded, axis + 1, repeats)
			result_shape = array_ops.concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
														axis=0)
			return array_ops.reshape(tiled, result_shape)

		# Broadcast the `repeats` tensor so rank(repeats) == axis + 1.
		if repeats.shape.ndims != axis + 1:
			repeats_shape = array_ops.shape(repeats)
			repeats_ndims = rank(repeats)
			broadcast_shape = array_ops.concat(
					[data_shape[:axis + 1 - repeats_ndims], repeats_shape], axis=0)
			repeats = gen_array_ops.broadcast_to(repeats, broadcast_shape)
			repeats.set_shape([None] * (axis + 1))

		# Create a "sequence mask" based on `repeats`, where slices across `axis`
		# contain one `True` value for each repetition.  E.g., if
		# `repeats = [3, 1, 2]`, then `mask = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]`.
		max_repeat = gen_math_ops.maximum(
				0, gen_math_ops._max(repeats, _all_dimensions(repeats)))
		mask = array_ops.sequence_mask(repeats, max_repeat)

		# Add a new dimension around each value that needs to be repeated, and
		# then tile that new dimension to match the maximum number of repetitions.
		expanded = array_ops.expand_dims(data, axis + 1)
		tiled = tile_one_dimension(expanded, axis + 1, max_repeat)

		# Use `boolean_mask` to discard the extra repeated values.  This also
		# flattens all dimensions up through `axis`.
		masked = array_ops.boolean_mask(tiled, mask)

		# Reshape the output tensor to add the outer dimensions back.
		if axis == 0:
			result = masked
		else:
			result_shape = array_ops.concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
														axis=0)
			result = array_ops.reshape(masked, result_shape)

		# Preserve shape information.
		if data.shape.ndims is not None:
			new_axis_size = 0 if repeats.shape[0] == 0 else None
			result.set_shape(data.shape[:axis].concatenate(
					[new_axis_size]).concatenate(data.shape[axis + 1:]))

		return result


def tile_one_dimension(data, axis, multiple):
	"""Tiles a single dimension of a tensor."""
	# Assumes axis is a nonnegative int.
	if data.shape.ndims is not None:
		multiples = [1] * data.shape.ndims
		multiples[axis] = multiple
	else:
		ones_value = array_ops.ones(array_ops.rank(data), dtypes.int32)
		multiples = array_ops.concat([ones_value[:axis], [multiple], ones_value[axis + 1:]],
											 axis=0)
	return array_ops.tile(data, multiples)


def _with_nonzero_rank(data):
	"""If `data` is scalar, then add a dimension; otherwise return as-is."""
	if data.shape.ndims is not None:
		if data.shape.ndims == 0:
			return array_ops.stack([data])
		else:
			return data
	else:
		data_shape = array_ops.shape(data)
		data_ndims = array_ops.rank(data)
		return array_ops.reshape(data, array_ops.concat([[1], data_shape], axis=0)[-data_ndims:])


def repeat(input, repeats, axis=None, name=None):  # pylint: disable=redefined-builtin
	"""Repeat elements of `input`.
	
	See also `tf.concat`, `tf.stack`, `tf.tile`.
	Args:
		input: An `N`-dimensional Tensor.
		repeats: An 1-D `int` Tensor. The number of repetitions for each element.
			repeats is broadcasted to fit the shape of the given axis. `len(repeats)`
			must equal `input.shape[axis]` if axis is not None.
		axis: An int. The axis along which to repeat values. By default (axis=None),
			use the flattened input array, and return a flat output array.
		name: A name for the operation.
	Returns:
		A Tensor which has the same shape as `input`, except along the given axis.
			If axis is None then the output array is flattened to match the flattened
			input array.
	Example usage:
	>>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
	<tf.Tensor: shape=(5,), dtype=string,
	numpy=array([b'a', b'a', b'a', b'c', b'c'], dtype=object)>
	>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
	<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
	array([[1, 2],
				 [1, 2],
				 [3, 4],
				 [3, 4],
				 [3, 4]], dtype=int32)>
	>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
	<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
	array([[1, 1, 2, 2, 2],
				 [3, 3, 4, 4, 4]], dtype=int32)>
	>>> repeat(3, repeats=4)
	<tf.Tensor: shape=(4,), dtype=int32, numpy=array([3, 3, 3, 3], dtype=int32)>
	>>> repeat([[1,2], [3,4]], repeats=2)
	<tf.Tensor: shape=(8,), dtype=int32,
	numpy=array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int32)>
	"""
	if axis is None:
		input = array_ops.reshape(input, [-1])
		axis = 0
	return repeat_with_axis(input, repeats, axis, name)