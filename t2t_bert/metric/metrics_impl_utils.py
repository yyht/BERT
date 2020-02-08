import tensorflow as tf
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops.metrics_impl import metric_variable
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops

def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
  """Calculate a streaming confusion matrix.
  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.
  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  # Local variable to accumulate the predictions in the confusion matrix.
  total_cm = metric_variable(
      [num_classes, num_classes], dtypes.float32, name='total_confusion_matrix')

  # Cast the type to int64 required by confusion_matrix_ops.
  predictions = math_ops.cast(predictions, dtypes.int32)
  labels = math_ops.cast(labels, dtypes.int32)
  num_classes = math_ops.cast(num_classes, dtypes.int32)

  # Flatten the input if its rank > 1.
  if predictions.get_shape().ndims > 1:
    predictions = array_ops.reshape(predictions, [-1])

  if labels.get_shape().ndims > 1:
    labels = array_ops.reshape(labels, [-1])

  if (weights is not None) and (weights.get_shape().ndims > 1):
    weights = array_ops.reshape(weights, [-1])

  # Accumulate the prediction to current confusion matrix.
  current_cm = confusion_matrix.confusion_matrix(
      labels, predictions, num_classes, weights=weights, dtype=dtypes.float32)
  update_op = state_ops.assign_add(total_cm, current_cm)
  return total_cm, update_op