# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Non-deterministic dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data_ops import random_ops
from utils.data_ops import dataset_ops as my_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from utils.data_ops import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from utils.data_ops import gen_experimental_dataset_ops
from utils.data_ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops

class _DirectedInterleaveDataset(dataset_ops.Dataset):
  """A substitute for `Dataset.interleave()` on a fixed list of datasets."""

  def __init__(self, selector_input, data_inputs):
    self._selector_input = selector_input
    self._data_inputs = list(data_inputs)

    first_output_types = my_dataset_ops.get_legacy_output_types(data_inputs[0])
    first_output_classes = my_dataset_ops.get_legacy_output_classes(data_inputs[0])

    for data_input in data_inputs[1:]:
      if (my_dataset_ops.get_legacy_output_types(data_input) != first_output_types
          or my_dataset_ops.get_legacy_output_classes(data_input)
          != first_output_classes):
        raise TypeError("All datasets must have the same type and class.")

    output_shapes = my_dataset_ops.get_legacy_output_shapes(self._data_inputs[0])
    for data_input in self._data_inputs[1:]:
      output_shapes = nest.pack_sequence_as(output_shapes, [
          ts1.most_specific_compatible_shape(ts2) for (ts1, ts2) in zip(
              nest.flatten(output_shapes),
              nest.flatten(my_dataset_ops.get_legacy_output_shapes(data_input)))
      ])

    self._structure = structure.convert_legacy_structure(
        first_output_types, output_shapes, first_output_classes)
    super(_DirectedInterleaveDataset, self).__init__()

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return (
        gen_experimental_dataset_ops.experimental_directed_interleave_dataset(
            self._selector_input._variant_tensor,
            [data_input._variant_tensor for data_input in self._data_inputs],
            **dataset_ops.flat_structure(self)))
    # pylint: enable=protected-access

  def _inputs(self):
    return [self._selector_input] + self._data_inputs

  @property
  def _element_structure(self):
    return self._structure

def sample_from_datasets(datasets, weights=None, seed=None):
  """Samples elements at random from the datasets in `datasets`.
  Args:
    datasets: A list of `tf.data.Dataset` objects with compatible structure.
    weights: (Optional.) A list of `len(datasets)` floating-point values where
      `weights[i]` represents the probability with which an element should be
      sampled from `datasets[i]`, or a `tf.data.Dataset` object where each
      element is such a list. Defaults to a uniform distribution across
      `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      random seed that will be used to create the distribution. See
      `tf.compat.v1.set_random_seed` for behavior.
  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.
  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError: If the `weights` argument is specified and does not match the
      length of the `datasets` element.
  """
  num_datasets = len(datasets)
  # Use each element of the given `weights` dataset as the probability of
  # choosing the respective input.

  # The `stateless_multinomial()` op expects log-probabilities, as opposed to
  # weights.
  logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))

  def select_dataset_varying_logits(logits, seed):
    return array_ops.squeeze(
        gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed),
        axis=[0, 1])

  logits_and_seeds = dataset_ops.Dataset.zip(
      (logits_ds, random_ops.RandomDataset(seed).batch(2)))
  selector_input = dataset_ops.MapDataset(
      logits_and_seeds,
      select_dataset_varying_logits,
      use_inter_op_parallelism=False)

  return _DirectedInterleaveDataset(selector_input, datasets)

def choose_from_datasets(datasets, choice_dataset):
  """Creates a dataset that deterministically chooses elements from `datasets`.
  For example, given the following datasets:
  ```python
  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
              tf.data.Dataset.from_tensors("bar").repeat(),
              tf.data.Dataset.from_tensors("baz").repeat()]
  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
  choice_dataset = tf.data.Dataset.range(3).repeat(3)
  result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
  ```
  The elements of `result` will be:
  ```
  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
  ```
  Args:
    datasets: A list of `tf.data.Dataset` objects with compatible structure.
    choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between
      `0` and `len(datasets) - 1`.
  Returns:
    A dataset that interleaves elements from `datasets` according to the values
    of `choice_dataset`.
  Raises:
    TypeError: If the `datasets` or `choice_dataset` arguments have the wrong
      type.
  """
  if not my_dataset_ops.get_structure(choice_dataset).is_compatible_with(
      structure.TensorStructure(dtypes.int64, [])):
    raise TypeError("`choice_dataset` must be a dataset of scalar "
                    "`tf.int64` tensors.")
  return _DirectedInterleaveDataset(choice_dataset, datasets)