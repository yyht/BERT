from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

from utils.bert import bert_utils

import tensorflow as tf

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.truediv(x, norm + epsilon)

def sort_key_val(t1, t2, dim=-1):
    values = tf.sort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, t1.shape)
    return values, tf.gather(t2, tf.argsort(t1, axis=dim), axis=dim)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))

def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor,  [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.chunk(tensor, chunks, axis=0)]
    return tf.reshape(torch.concat(summed_tensors, axis=0), orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class ScaleNorm(object):
    def __init__(self, emb, eps):
        super(ScaleNorm, self).__init__()
        self.g = tf.Variable(initial_value=w_init(shape=(1,),
                        dtype='float32'),
                        trainable=True)
        self.eps = eps

    def __call__(self, inputs):
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return inputs / n * self.g

class WithNorm(object):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def __call__(self, inputs):
        inputs = self.norm(inputs)
        return self.fn(inputs)

class Chunk(object):
    def __init__(self, chunks, fn, along_axis = -1):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def __call__(self, inputs):
        chunks = tf.split(inputs, self.chunks, axis= self.axis)
        return tf.concat([self.fn(c) for c in chunks], axis = self.axis)