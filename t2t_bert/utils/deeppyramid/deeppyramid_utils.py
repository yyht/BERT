import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.util import nest

import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)



def dpcnn_two_layers_conv(config, inputs, is_training_flag, 
                        double_num_filters=False):
    """
    two layers of conv
    inputs:[batch_size,total_sequence_length,embed_size,dimension]. e.g.(128, 400, 64,1)-->[128,200,32,250]
    :return:[batch_size,total_sequence_length,embed_size,num_filters]
    """
    # conv1:
    # filter1's first three dimension apply to [total_sequence_length, embed_size, 1] of embedding_documents
    print("dpcnn_two_layers_conv.inputs:", inputs)  # (128, 400, 64, 250)
    channel = inputs.get_shape().as_list()[-1]
    if double_num_filters:
        hpcnn_number_filters = channel * 2
    else:
        hpcnn_number_filters = config.hpcnn_number_filters
    filter1 = tf.get_variable("filter1-%s" % config.hpcnn_filter_size,[config.hpcnn_filter_size, 1, channel, hpcnn_number_filters],initializer=initializer)
    conv1 = tf.nn.conv2d(inputs, filter1, strides=[1, config.stride_length, 1, 1], padding="SAME",name="conv")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_training_flag, scope='cnn1')

    print("dpcnn_two_layers_conv.conv1:", conv1)  # (128, 400, 64, 250)
    b1 = tf.get_variable("b-cnn-%s" % hpcnn_number_filters, [hpcnn_number_filters])
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1),"relu1")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]

    # conv2
    # filter2's first three dimension apply to:[total_sequence_length,embed_size,hpcnn_number_filters] of conv1
    filter2 = tf.get_variable("filter2-%s" % config.hpcnn_filter_size,[config.hpcnn_filter_size, 1, hpcnn_number_filters, hpcnn_number_filters],initializer=initializer)
    conv2 = tf.nn.conv2d(conv1, filter2, strides=[1, config.stride_length, 1, 1], padding="SAME",name="conv2")  # shape:[batch_size,stotal_sequence_length,embed_size,hpcnn_number_filters]
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training_flag, scope='cnn2')

    print("dpcnn_two_layers_conv.conv2:", conv2)  # (128, 400, 64, 250)
    return conv2  # shape:[batch_size,total_sequence_length,embed_size,num_filters]

def dpcnn_pooling_two_conv(config, conv, layer_index, is_training_flag):
    """
    pooling followed with two layers of conv, used by deep pyramid cnn.
    pooling-->conv-->conv-->skip connection
    conv:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
    :return:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
    """
    with tf.variable_scope("pooling_two_conv_" + str(layer_index)):
        # 1. pooling:max-pooling with size 3 and stride 2==>reduce shape to half
        pooling = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name="pool")  # [batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
        print(layer_index, "dpcnn_pooling_two_conv.pooling:", pooling)

        # 2. two layer of conv
        conv = dpcnn_two_layers_conv(config, pooling, is_training_flag, 
                                    double_num_filters=False) #TODO double num_filters
        # print("dpcnn_pooling_two_conv.layer_index", layer_index, "conv:", conv)

        # 3. skip connection and activation
        conv = conv + pooling
        b = tf.get_variable("b-poolcnn%s" % config.hpcnn_number_filters, [config.hpcnn_number_filters])
        conv = tf.nn.relu(tf.nn.bias_add(conv, b),"relu-poolcnn")  # shape:[batch_size,total_sequence_length/2,embed_size/2,hpcnn_number_filters]
    return conv

def deep_pyramid_cnn(config, embedding_documents, is_training_flag):
    """
    deep pyramid cnn for text categorization
    region embedding-->two layers of convs-->repeat of building block(Pooling,/2-->conv-->conv)--->pooling
    for more check: http://www.aclweb.org/anthology/P/P17/P17-1052.pdf
    :return: logits_list
    """
    # 2.two layers of convs
    embedding_documents = tf.expand_dims(embedding_documents, -1)  # [batch_size,total_sequence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
    conv = dpcnn_two_layers_conv(config, embedding_documents, is_training_flag)  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
    # 2.1 skip connection: add and activation
    conv = conv + embedding_documents  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]
    b = tf.get_variable("b-inference", [config.hpcnn_number_filters])
    conv = tf.nn.relu(tf.nn.bias_add(conv, b), "relu-inference")  # shape:[batch_size,total_sequence_length,embed_size,hpcnn_number_filters]

    # 3.repeat of building blocks
    for i in range(config.num_repeat):
        conv = dpcnn_pooling_two_conv(config, conv, i, is_training_flag)  # shape:[batch_size,total_sequence_length/np.power(2,i),hpcnn_number_filters]

    # 4.max pooling
    seq_length1 = conv.get_shape().as_list()[1]  # sequence length after multiple layers of conv and pooling
    seq_length2 = conv.get_shape().as_list()[2]  # sequence length after multiple layers of conv and pooling
    print("before.final.pooling:", conv) #(256, 25, 4, 16)
    pooling = tf.nn.max_pool(conv, ksize=[1, seq_length1, seq_length2, 1], strides=[1, 1, 1, 1], padding='VALID',name="pool")  # [batch_size,hpcnn_number_filters]
    pooling = tf.squeeze(pooling) #(256, 16)
    print("pooling.final:", pooling)

    return pooling

def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size = None):
    with tf.variable_scope(scope or "highway_layer"):
        if output_size is not None:
            d = output_size
        else:
            d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)

        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        if d != arg.get_shape()[-1]:
            arg = linear([arg], d, bias, bias_start=bias_start, scope='arg_resize', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size = None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train, output_size = output_size)
            prev = cur
        return cur

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    with tf.variable_scope(scope or "linear"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        flat_args = [flatten(arg, 1) for arg in args]
        # if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                         for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias)
        out = reconstruct(flat_out, args[0], 1)
        if squeeze:
            out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    return out

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    # print("out shape")
    # print(out.get_shape())
    return out