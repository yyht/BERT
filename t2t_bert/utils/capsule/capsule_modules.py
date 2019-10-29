from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from utils.capsule import capsule_utils
from utils.capsule import capsule_layers

def baseline_model_cnn(X, num_classes):
    nets = capsule_utils._conv2d_wrapper(
        X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
        add_bias=False, activation_fn=tf.nn.relu, name='conv1'
        )
    nets = tf.layers.flatten(nets)
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))
    nets = tf.layers.dense(nets, 128, scope='relu_fc3', activation=tf.nn.relu)
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))
    
    activations = tf.nn.sigmoid(tf.layers.dense(nets, num_classes, scope='final_layer', activation=None))
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))
    return tf.zeros([0]), activations

def baseline_model_kimcnn(X, max_sent, num_classes):
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")            
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.nn.sigmoid(tf.layers.dense(h_pool_flat, num_classes, scope='final_layer', activation=None))
    return tf.zeros([0]), activations
        
def capsule_model_B(X, num_classes):
    poses_list = []
    for _, ngram in enumerate([3,4,5]):
        with tf.variable_scope('capsule_'+str(ngram)): 
            nets = capsule_utils._conv2d_wrapper(
                X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsule_layers.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
            nets = capsule_layers.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = capsule_layers.capsule_flatten(nets)
            poses, activations = capsule_layers.capsule_fc_layer(nets, num_classes, 3, 'fc2')
            poses_list.append(poses)
    
    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0) 
    activations = tf.sqrt(tf.reduce_sum(tf.square(poses), 2))
    return poses, activations

def capsule_model_A(X, num_classes):
    with tf.variable_scope('capsule_'+str(3)):   
        nets = capsule_utils._conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsule_layers.capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1], 
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')                        
        nets = capsule_layers.capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_layers.capsule_flatten(nets)
        poses, activations = capsule_layers.capsule_fc_layer(nets, num_classes, 3, 'fc2') 
    return poses, activations