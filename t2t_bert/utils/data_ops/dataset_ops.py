from utils.data_ops import structure as structure_lib


def get_structure(dataset_or_iterator):
  """Returns the `tf.data.experimental.Structure` of a `Dataset` or `Iterator`.
  Args:
    dataset_or_iterator: A `tf.data.Dataset`, `tf.compat.v1.data.Iterator`, or
      `IteratorV2`.
  Returns:
    A `tf.data.experimental.Structure` representing the structure of the
    elements of `dataset_or_iterator`.
  Raises:
    TypeError: If `dataset_or_iterator` is not a dataset or iterator object.
  """
  try:
    ret = dataset_or_iterator._element_structure  # pylint: disable=protected-access
    if isinstance(ret, structure_lib.Structure):
      return ret
  except AttributeError:
    pass
  raise TypeError("`dataset_or_iterator` must be a Dataset or Iterator object, "
                  "but got %s." % type(dataset_or_iterator))

def get_legacy_output_shapes(dataset_or_iterator):
  """Returns the output shapes of a `Dataset` or `Iterator`.
  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_shapes` property.
  Args:
    dataset_or_iterator: A `tf.data.Dataset`, `tf.compat.v1.data.Iterator`, or
      `IteratorV2`.
  Returns:
    A nested structure of `tf.TensorShape` objects corresponding to each
    component of an element of the given dataset or iterator.
  """
  return get_structure(dataset_or_iterator)._to_legacy_output_shapes()  # pylint: disable=protected-access

def get_legacy_output_types(dataset_or_iterator):
  """Returns the output shapes of a `Dataset` or `Iterator`.
  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_types` property.
  Args:
    dataset_or_iterator: A `tf.data.Dataset`, `tf.compat.v1.data.Iterator`, or
      `IteratorV2`.
  Returns:
    A nested structure of `tf.DType` objects corresponding to each component
    of an element of this dataset.
  """
  return get_structure(dataset_or_iterator)._to_legacy_output_types()  # pylint: disable=protected-access

def get_legacy_output_classes(dataset_or_iterator):
  """Returns the output classes of a `Dataset` or `Iterator`.
  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_classes` property.
  Args:
    dataset_or_iterator: A `tf.data.Dataset`, `tf.compat.v1.data.Iterator`, or
      `IteratorV2`.
  Returns:
    A nested structure of Python `type` or `tf.data.experimental.Structure`
    objects corresponding to each component of an element of this dataset.
  """
  return get_structure(dataset_or_iterator)._to_legacy_output_classes()  # pylint: disable=protected-access