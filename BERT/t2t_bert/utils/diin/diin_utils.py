import tensorflow as tf
from model.utils.diin.util import blocks
from model.utils.diin.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, linear, conv2d, cosine_similarity, variable_summaries, dense_logits, fuse_gate
from model.utils.diin.tensorflow import flatten, reconstruct, add_wd, exp_mask
import numpy as np


def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "dense_logit_bi_attention"):
        PL = p.get_shape()[1]
        HL = h.get_shape()[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,config.max_seq_len_word,1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1,config.max_seq_len_word,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, config.max_seq_len_word, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, config.max_seq_len_word, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug
        ph_mask = None

        
        h_logits = p_aug * h_aug
        
        return h_logits


def self_attention(config, is_train, p, p_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape()[1]
        dim = p.get_shape()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,config.max_seq_len_word,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,config.max_seq_len_word,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, config.max_seq_len_word, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, config.max_seq_len_word, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2


        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits) 

        return self_att


def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(config, is_train, p, p_mask=p_mask)

        print("self_att shape")
        print(self_att.get_shape())
        
        p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")
        
        return p0




# def bi_attention(config, is_train, p, h, p_mask=None, h_mask=None, scope=None, h_value = None): #[N, L, 2d]
#     with tf.variable_scope(scope or "bi_attention"):
#         PL = tf.shape(p)[1]
#         HL = tf.shape(h)[1]
#         p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
#         h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]


#         if p_mask is None:
#             ph_mask = None
#         else:
#             p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
#             h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
#             ph_mask = p_mask_aug & h_mask_aug


#         h_logits = get_logits([p_aug, h_aug], None, True, wd=config.wd, mask=ph_mask,
#                           is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
#         h_a = softsel(h_aug, h_logits) 
#         p_a = softsel(p, tf.reduce_max(h_logits, 2))  # [N, 2d]
#         p_a = tf.tile(tf.expand_dims(p_a, 1), [1, PL, 1]) # 

#         return h_a, p_a


def dense_net(config, denseAttention, is_train):
    with tf.variable_scope("dense_net"):
        
        dim = denseAttention.get_shape().as_list()[-1]
        act = tf.nn.relu if config.first_scale_down_layer_relu else None
        fm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding="SAME", activation_fn = act)

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "first_dense_net_block") 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "second_dense_net_block") 
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train ,scope = "third_dense_net_block") 

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(fm, [-1, shape_list[1]*shape_list[2]*shape_list[3]])
        return out_final



def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding="SAME", act=tf.nn.relu, scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):        
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features 

def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):


        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn = None)
        
        
        feature_map = tf.nn.max_pool(feature_map, [1,2,2,1],[1,2,2,1], "VALID")

        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map