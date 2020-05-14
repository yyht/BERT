from utils.textcnn import conv1d_transpose as my_conv1d_transpose
from utils.bert import bert_utils

def deconv_op(inputs,
			filters,
			kernel_size,
			output_shape,
			padding="same",
			activation=None,
			strides=1,
			reuse=None,
			name="deconv",
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
			biase_initializer=tf.zeros_initializer()):
	
	"""
	input: A 3-D `Tensor` of type `float` and shape
	  `[batch, in_width, in_channels]` for `NWC` data format or
	  `[batch, in_channels, in_width]` for `NCW` data format.
	filters: A 3-D `Tensor` with the same type as `value` and shape
	  `[filter_width, output_channels, in_channels]`.  `filter`'s
	  `in_channels` dimension must match that of `value`.
	output_shape: A 1-D `Tensor`, containing three elements, representing the
	  output shape of the deconvolution op.
	"""

	input_shape = bert_utils.get_shape_list(inputs, expected_rank=[3])

	filter_width, output_channels, in_channels
	filter_shape = [kernel_size, filters, input_shape[-1]]

	W = tf.get_variable("%s_W"%name, filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))

	output = my_conv1d_transpose.conv1d_transpose(
			inputs,  # pylint: disable=redefined-builtin
			W,
			output_shape, # must be valid shape
			strides,
			padding=padding,
			data_format="NWC",
			dilations=None,
			name=None)

	return output