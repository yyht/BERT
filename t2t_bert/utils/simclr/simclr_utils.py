from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

def linear_layer(x,
								 is_training,
								 num_classes,
								 use_bias=True,
								 use_bn=False,
								 name='linear_layer'):
	"""Linear head for linear evaluation.
	Args:
		x: hidden state tensor of shape (bsz, dim).
		is_training: boolean indicator for training or test.
		num_classes: number of classes.
		use_bias: whether or not to use bias.
		use_bn: whether or not to use BN for output units.
		name: the name for variable scope.
	Returns:
		logits of shape (bsz, num_classes)
	"""
	assert x.shape.ndims == 2, x.shape
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		x = tf.layers.dense(
				inputs=x,
				units=num_classes,
				use_bias=use_bias and not use_bn,
				kernel_initializer=tf.random_normal_initializer(stddev=.01))
		x = tf.identity(x, '%s_out' % name)
	return x

def projection_head(hiddens, is_training, 
									head_proj_dim=128,
									num_nlh_layers=1,
									head_proj_mode='nonlinear',
									name='head_contrastive'):
	"""Head for projecting hiddens fo contrastive loss."""
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		if head_proj_mode == 'none':
			pass  # directly use the output hiddens as hiddens
		elif head_proj_mode == 'linear':
			hiddens = linear_layer(
					hiddens, is_training, head_proj_dim,
					use_bias=False, use_bn=True, name='l_0')
		elif head_proj_mode == 'nonlinear':
			hiddens = linear_layer(
					hiddens, is_training, hiddens.shape[-1],
					use_bias=True, use_bn=True, name='nl_0')
			for j in range(1, num_nlh_layers + 1):
				hiddens = tf.nn.relu(hiddens)
				hiddens = linear_layer(
						hiddens, is_training, head_proj_dim,
						use_bias=False, use_bn=True, name='nl_%d'%j)
		else:
			raise ValueError('Unknown head projection mode {}'.format(head_proj_mode))
	return hiddens

