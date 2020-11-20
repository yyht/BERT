from utils.bert import bert_utils
from utils.bert import bert_modules

import numpy as np

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
from loss import loss_utils
from task_module.global_batch_norm import batch_norm_relu 

try:
  from task_module.contrastive_utils import add_contrastive_loss as simlr_contrastive_loss_fn
  tf.logging.info("** export multi-tpu simclr **")
except:
  simlr_contrastive_loss_fn = None
  tf.logging.info("** using single-tpu simclr **")

"""
https://github.com/google-research/mixmatch/blob/master/mixup.py
https://github.com/google-research/simclr/blob/master/model.py
"""
LARGE_NUM = 1e9

def linear_layer(FLAGS, x,
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
    if use_bn:
      x = batch_norm_relu(FLAGS, x, is_training, relu=False, center=use_bias)
    x = tf.identity(x, '%s_out' % name)
  return x

def projection_head(FLAGS, hiddens, is_training, 
                use_bn=False,
                name='head_contrastive'):
  """Head for projecting hiddens fo contrastive loss."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    mid_dim = hiddens.shape[-1]
    out_dim = FLAGS.proj_out_dim
    hiddens_list = [hiddens]
    if FLAGS.proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens.
    elif FLAGS.proj_head_mode == 'linear':
      hiddens = linear_layer(FLAGS,
          hiddens, is_training, out_dim,
          use_bias=False, use_bn=use_bn, name='l_0')
      hiddens_list.append(hiddens)
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          dim, bias_relu = mid_dim, True
        else:
          # for the final layer, neither bias nor relu is used.
          dim, bias_relu = FLAGS.proj_out_dim, False
        hiddens = linear_layer(FLAGS,
            hiddens, is_training, dim,
            use_bias=bias_relu, use_bn=use_bn, name='nl_%d'%j)
        hiddens = tf.nn.relu(hiddens) if bias_relu else hiddens
        hiddens_list.append(hiddens)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    # take the projection head output during pre-training.
    hiddens = hiddens_list[-1]
  return hiddens

def _sample_from_softmax(logits, disallow=None):
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(
     bert_utils.get_shape_list(logits), minval=1e-10, maxval=1-1e-10)
  gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-10) + 1e-10)
  return tf.one_hot(tf.argmax(tf.nn.softmax(logits + gumbel_noise), -1,
                              output_type=tf.int32), logits.shape[-1])

def _sample_positive(features, batch_size):

  logits = tf.log(1./tf.ones((batch_size, batch_size), dtype=tf.float32)/batch_size)
  # [batch_size, batch_size]
  disallow_mask = tf.one_hot(tf.range(batch_size), batch_size)

  sampled_onthot = _sample_from_softmax(logits, disallow_mask)
  positive_ids = tf.argmax(sampled_onthot, axis=-1, output_type=tf.int32)
  # sampled_feature = tf.gather_nd(features, positive_ids[:, None])

  # batch_idx = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int32), axis=-1)
  # gather_index = tf.concat([batch_idx, positive_ids[:, None]], axis=-1)
  # sampled_feature = tf.gather_nd(features, gather_index)
  sampled_feature = tf.gather_nd(features, positive_ids[:, None])
  return sampled_feature, positive_ids

def my_contrastive_loss(hidden,
                     hidden_norm=True,
                     temperature=1.0,
                     tpu_context=None,
                     weights=1.0):

    if hidden_norm:
      hidden = tf.nn.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)

    batch_size = bert_utils.get_shape_list(hidden1, expected_rank=[2,3])[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
    loss = loss_a + loss_b

    tf.logging.info(hidden1)
    tf.logging.info(hidden2)
    tf.logging.info(logits_ab)
    tf.logging.info(logits_ba)
    tf.logging.info(labels)
    tf.logging.info(logits_aa)
    tf.logging.info(logits_bb)
    tf.logging.info(hidden1_large)
    tf.logging.info(hidden2_large)

    return loss, logits_ab, labels

def linear_mixup(hidden, sampled_hidden, beta=0.5):
    hidden_shape_list = bert_utils.get_shape_list(hidden, expected_rank=[2,3])
    batch_size = hidden_shape_list[0]
    hidden_dims = hidden_shape_list[-1]

    uniform_noise = tf.random.uniform([batch_size, 1], minval=0.9, maxval=1)
    mix = tf.cast(tf.maximum(uniform_noise, 1 - uniform_noise), tf.float32)

    tf.logging.info(hidden)
    tf.logging.info(sampled_hidden)

    xmix_linear = hidden * mix + sampled_hidden * (1.0 - mix)
    return xmix_linear

def binary_mixup(hidden, sampled_hidden, beta=0.5):
    hidden_shape_list = bert_utils.get_shape_list(hidden, expected_rank=[2,3])
    batch_size = hidden_shape_list[0]
    hidden_dims = hidden_shape_list[-1]

    binary_noise = tf.random.uniform([batch_size, hidden_dims], minval=0.7, maxval=1)
    binary_noise_dist = tf.distributions.Bernoulli(probs=binary_noise, 
                                                dtype=tf.float32)
    binary_mask = binary_noise_dist.sample()
    binary_mask = tf.cast(binary_mask, tf.float32)
    xmix_binary = hidden * binary_mask +  sampled_hidden * (1.0 - binary_mask)
    return xmix_binary

def random_mixup_modified(hidden, sampled_hidden_a, sampled_hidden_b, beta=0.5):
    linear_feat_a = linear_mixup(hidden, sampled_hidden_a, beta)
    binary_feat_a = binary_mixup(hidden, sampled_hidden_a, beta)

    linear_feat_b = linear_mixup(hidden, sampled_hidden_b, beta)
    binary_feat_b = binary_mixup(hidden, sampled_hidden_b, beta)

    # [batch_size, 2_ab, hidden_dims]
    linear_mixup_noise = tf.stack([linear_feat_a, linear_feat_b], axis=1)
    binary_mixup_noise = tf.stack([binary_feat_a, binary_feat_b], axis=1)

    tf.logging.info(linear_mixup_noise)
    tf.logging.info(binary_mixup_noise)

    # [batch_size, 2_linear_binary, 2_ab, hidden_dims]
    mixup_matrix = tf.stack([linear_mixup_noise, binary_mixup_noise], axis=1)

    tf.logging.info(mixup_matrix)

    mixup_matrix_shape = bert_utils.get_shape_list(mixup_matrix, expected_rank=[2,3])

    batch_size = mixup_matrix_shape[0]
    noise_num = mixup_matrix_shape[1]

    sample_prob = tf.ones((batch_size, noise_num), dtype=tf.float32)/noise_num
    mixup_noise_idx = tf.multinomial(tf.log(sample_prob)+1e-10,
              num_samples=1,
              output_dtype=tf.int32) # batch x 1

    batch_idx = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int32), axis=-1)
    gather_index = tf.concat([batch_idx, mixup_noise_idx], axis=-1)

    mixup_noise = tf.gather_nd(mixup_matrix, gather_index)

    noise_lst = tf.unstack(mixup_noise, axis=1)
    xmix_features = tf.concat(noise_lst, axis=0)
    return xmix_features

def random_mixup(hidden, sampled_hidden, beta=0.5):

    hidden_shape_list = bert_utils.get_shape_list(hidden, expected_rank=[2,3])
    batch_size = hidden_shape_list[0]
    hidden_dims = hidden_shape_list[-1]

    # mix = tf.distributions.Beta(beta, beta).sample([batch_size, 1])
    uniform_noise = tf.random.uniform([batch_size, 1], minval=0.9, maxval=1)
    mix = tf.cast(tf.maximum(uniform_noise, 1 - uniform_noise), tf.float32)

    tf.logging.info(hidden)
    tf.logging.info(sampled_hidden)

    xmix_linear = hidden * mix + sampled_hidden * (1.0 - mix)
    # xmix_geometric = tf.pow(hidden, mix) * tf.pow(sampled_hidden, (1.0 - mix))

    binary_noise = tf.random.uniform([batch_size, hidden_dims], minval=0.7, maxval=1)
    binary_noise_dist = tf.distributions.Bernoulli(probs=binary_noise, 
                                                dtype=tf.float32)
    binary_mask = binary_noise_dist.sample()
    binary_mask = tf.cast(binary_mask, tf.float32)
    xmix_binary = hidden * binary_mask +  sampled_hidden * (1.0 - binary_mask)

    # mixup_noise_sample = [xmix_linear, xmix_geometric, xmix_binary]
    mixup_noise_sample = [xmix_linear, xmix_binary]
    # [batch_size, len(mixup_noise_sample), hidden_dims]
    mixup_matrix = tf.stack(mixup_noise_sample, axis=1)

    mixup_matrix_shape = bert_utils.get_shape_list(mixup_matrix, expected_rank=[2,3])

    batch_size = mixup_matrix_shape[0]
    noise_num = mixup_matrix_shape[1]

    sample_prob = tf.ones((batch_size, noise_num), dtype=tf.float32)/noise_num
    mixup_noise_idx = tf.multinomial(tf.log(sample_prob)+1e-10,
              num_samples=1,
              output_dtype=tf.int32) # batch x 1

    batch_idx = tf.expand_dims(tf.cast(tf.range(batch_size), tf.int32), axis=-1)
    gather_index = tf.concat([batch_idx, mixup_noise_idx], axis=-1)
    mixup_noise = tf.gather_nd(mixup_matrix, gather_index)
    return mixup_noise

def mixup_dsal_plus(config, 
        hidden,
        input_mask,
        temperature=0.1,
        hidden_norm=True,
        masked_repres=None,
        is_training=True,
        beta=0.5,
        use_bn=True,
        tpu_context=None,
        weights=1.0,
        sent_repres_mode='cls',
        negative_mode='local',
        monitor_dict={}):
    input_shape_list = bert_utils.get_shape_list(hidden, expected_rank=[2, 3])
    batch_size = input_shape_list[0]
    hidden_dims = input_shape_list[-1]

    if sent_repres_mode == 'cls':
      sent_repres = tf.squeeze(tf.identity(hidden[:, 0:1, :]))
      tf.logging.info("== apply cls-sent_repres ==")
    elif sent_repres_mode == 'mean_pooling':
      hidden_mask = tf.cast(input_mask[:, :, None], dtype=tf.float32)
      sent_repres = tf.reduce_sum(hidden_mask*hidden, axis=1)
      sent_repres /= (1e-10 + tf.reduce_sum(hidden_mask, axis=1))
      tf.logging.info("== apply mean-pooling-sent_repres ==")

    # [batch_size, hidden_dims]
    # [positive_1_repres, 
    # positive_1_ids] = _sample_positive(sent_repres, batch_size)
    # xmix_a = random_mixup(sent_repres, positive_1_repres, beta=beta)
    
    # [positive_2_repres, 
    # positive_2_ids] = _sample_positive(sent_repres, batch_size)
    # xmix_b = random_mixup(sent_repres, positive_2_repres, beta=beta)

    # xmix_features = tf.concat([xmix_a, xmix_b], 0)  # (num_transforms * bsz, h, w, c)

    # [batch_size, hidden_dims]
    [positive_1_repres, 
    positive_1_ids] = _sample_positive(sent_repres, batch_size)

    [positive_2_repres, 
    positive_2_ids] = _sample_positive(sent_repres, batch_size)
    xmix_features = random_mixup_modified(sent_repres, positive_1_repres, positive_2_repres, beta=beta)

    with tf.variable_scope('cls/simclr_projection_head', reuse=tf.AUTO_REUSE):
      xmix_hiddens = projection_head(config, 
                              xmix_features, 
                              is_training, 
                              name='head_contrastive')

    # [2*batch_size, hidden_dims]
    if simlr_contrastive_loss_fn:
      if negative_mode == 'local':
        contrastive_loss_fn = my_contrastive_loss
        tf.logging.info("== apply tpu-simclr local batch ==")
      else:
        contrastive_loss_fn = simlr_contrastive_loss_fn
        tf.logging.info("== apply tpu-simclr cross batch ==")
    else:
      contrastive_loss_fn = my_contrastive_loss
      tf.logging.info("== apply simclr local batch ==")
    [contrast_loss, logits_con, labels_con] = contrastive_loss_fn(xmix_hiddens,
                     hidden_norm=hidden_norm,
                     temperature=temperature,
                     tpu_context=tpu_context,
                     weights=weights)

    contrast_acc = tf.equal(tf.argmax(labels_con, 1), tf.argmax(logits_con, axis=1))
    contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))

    monitor_dict['contrast_acc'] = contrast_acc

    return contrast_loss, monitor_dict
