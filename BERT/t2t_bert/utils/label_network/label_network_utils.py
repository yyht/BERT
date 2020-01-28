import tensorflow as tf
import numpy as np
import math
# from tensor2tensor.models.research import universal_transformer, universal_transformer_util
import tensorflow as tf
import numpy as np
from utils.qanet import qanet_layers

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)



def hard_attention_mask(presence_vec, threshold):
    init_condition = tf.zeros_like(presence_vec)
    threshold_vec = tf.ones_like(presence_vec) * threshold
    condition = tf.less_equal(presence_vec, threshold_vec)

    default_values = tf.ones_like(presence_vec)
    mask = tf.where(condition, init_condition, default_values)
    mask = tf.cast(mask, tf.float32)
    return mask

def _position_encoding(position_size, dim, 
                    min_timescale=1.0,
                    max_timescale=1.0e4):
    position = tf.to_float(tf.range(position_size))
    num_timescales = dim // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) \
        * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(dim, 2)]])
    signal = tf.reshape(signal, [1, position_size, dim])

    return signal

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)

def memory_attention(query, memory, 
                    query_mask, scope, 
                    memory_mask=None,
                    reuse=None):
    """
    query: batch x len x query_dim
    memory: batch x num_calsses x mem_dim
    """
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):

        query_dim = query.get_shape()[-1]
        mem_dim = memory.get_shape()[-1]

        # batch x num_calsses x mem_dim
        memory_ = tf.transpose(memory, [0,2,1])

        attn_W = tf.get_variable("AttnW", dtype=tf.float32,
                                    shape=[query_dim, mem_dim],
                                    initializer=initializer)

        # bacth x len x mem_dim
        weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]])
        
        # batch x len x num_classes
        S = tf.matmul(weighted_query, memory_)

        # batch x 1 x len
        mask_q = tf.expand_dims(query_mask, axis=1)

        # batch x num_classes x len
        S_ = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        # batch x num_classes x dim
        output = tf.matmul(S_, query)
        print(output.get_shape(), "=====")

        output = tf.reduce_sum(output, axis=1)

        return output

def memory_attention_v1(query, memory, 
                    query_mask, scope,
                    memory_mask=None,
                    reuse=None,
                    attention_output="soft",
                    num_heads=8,
                    dropout_rate=0.0,
                    threshold=0.1,
                    apply_hard_attn=False):
    """
    query: batch x len x query_dim
    memory: batch x num_calsses x mem_dim
    """
    with tf.variable_scope(scope+"_label_attention", reuse=reuse):

        query_dim = query.get_shape()[-1]
        mem_dim = memory.get_shape()[-1]

        # batch x num_calsses x mem_dim
        memory_ = tf.transpose(memory, [0,2,1])

        attn_W = tf.get_variable("AttnW", dtype=tf.float32,
                                    shape=[query_dim, mem_dim],
                                    initializer=initializer)

        # bacth x len x mem_dim
        weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]])
        
        # batch x len x num_classes
        S = tf.matmul(weighted_query, memory_)

        # batch x 1 x len
        mask_q = tf.expand_dims(query_mask, axis=1)

        # batch x num_classes x len
        S_ = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        # batch x num_classes x dim
        output = tf.matmul(S_, query)
        print(output.get_shape(), "==output shape===")
        if apply_hard_attn:
            presence_vec = output * output # batch x num_class x dim
            presence_vec = tf.sqrt(tf.reduce_sum(presence_vec, axis=-1)) # batch x num_class

            presence_vec = tf.nn.softmax(presence_vec, axis=-1)
            presence_mask = hard_attention_mask(
                            presence_vec,
                            threshold)

            output *= tf.expand_dims(presence_mask, -1)

            # presence_vec = tf.nn.softmax(presence_vec)
            # idx = tf.where(presence_vec > threshold)
            # batch_idxs = tf.range(0, tf.shape(output)[0])
            # batch_idxs = tf.expand_dims(batch_idxs, 1)

            # idxs = tf.concat([batch_idxs, idx], 1)
            # output = tf.gather_nd(output, idxs)

            print(output.get_shape(), "==hard attention output shape===")

        if attention_output == "soft" and not apply_hard_attn:
            class_dim = memory.get_shape()[1]
            class_attention = tf.get_variable("class_attn", dtype=tf.float32,
                                    shape=[query_dim],
                                    initializer=initializer)
            # batch x num_classes
            attn_output = tf.reduce_sum(output * class_attention,
                                axis=-1) 
            attn_output = tf.softmax(attn_output) # batch x num_classes
            attn_output = tf.expand_dims(attn_output, axis=-1) # batch x num_classes x 1
            output = tf.reduce_sum(attn_output * output, axis=1)

        elif attention_output == "sum" and apply_hard_attn:
            output = tf.reduce_sum(output, axis=1)

        elif attention_output == "multi_head":
            # get memory mask
            ignore_padding = (1 - presence_mask)
            ignore_padding = attention_bias_ignore_padding(ignore_padding)
            encoder_self_attention_bias = ignore_padding

            output = multihead_attention_texar(output, 
                            memory=None, 
                            memory_attention_bias=encoder_self_attention_bias,
                            num_heads=num_heads, 
                            num_units=None, 
                            dropout_rate=dropout_rate, 
                            scope="multihead_attention")
            output = tf.reduce_sum(output, axis=1)

        return output

def highway_layer(in_val, scope=None):
    output_size = in_val.get_shape()[-1]
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    return outputs

def multi_highway_layer(in_val, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, scope=cur_scope_name)
    return in_val

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len,
        memory_length
    """
    ret = memory_padding * -1e18
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def _split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads,
        becomes dimension 1).
    Must ensure `x.shape[-1]` can be deviced by num_heads
    """
    depth = x.get_shape()[-1]
    print(x.get_shape(), "===splitheads===")
    splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
        num_heads, depth // num_heads])
    return tf.transpose(splitted_x, [0, 2, 1, 3])

def _combine_heads(x):
    """
    Args:
        x: A Tensor of shape `[batch, num_heads, seq_len, dim]`

    Returns:
        A Tensor of shape `[batch, seq_len, num_heads * dim]`
    """
    t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
    num_heads, dim = t.get_shape()[-2:]
    return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])

def multihead_attention_texar(queries, 
                memory=None, 
                memory_attention_bias=None,
                num_heads=8, 
                num_units=None, 
                dropout_rate=0.0, 
                scope="multihead_attention"):
    if num_units is None:
        num_units = queries.get_shape()[-1]
    if num_units % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the"
                             "number of attention heads (%d)." % (\
                            num_units, num_heads))
    if memory is None:
        Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
        K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
        V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
    else:
        Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
        K = tf.layers.dense(memory, num_units, use_bias=False, name='k')
        V = tf.layers.dense(memory, num_units, use_bias=False, name='v')

    Q_ = _split_heads(Q, num_heads)
    K_ = _split_heads(K, num_heads)
    V_ = _split_heads(V, num_heads)

    key_depth_per_head = num_units // num_heads
    Q_ *= tf.pow(tf.cast(key_depth_per_head, tf.float32), -0.5)

    logits = tf.matmul(Q_, K_, transpose_b=True)
    if memory_attention_bias is not None:
        logits += memory_attention_bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1 - dropout_rate)
    outputs = tf.matmul(weights, V_)

    outputs = _combine_heads(outputs)
    outputs = tf.layers.dense(outputs, num_units,\
            use_bias=False, name='output_transform')
        #(batch_size, length_query, attention_depth)
    return outputs

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked. 
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
    Returns
        A 3d tensor with shape of (N, T_q, C)   
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
            
        # Dropouts
        outputs = tf.nn.dropout(outputs, dropout_rate)
                     
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                    
        # Residual connection
        outputs += queries
                    
        # Normalize
        outputs = tf.contrib.layers.layer_norm(outputs)

        # outputs = normalize(outputs) # (N, T_q, C)
        outputs = tf.cast(outputs, dtype=tf.float32)

    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
    Returns:
        A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                            "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readinner layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 5,
                            "activation": tf.nn.relu, "use_bias": True, "padding":"same"}
        outputs = tf.layers.conv1d(**params)


        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                            "activation": None, "use_bias": True, "padding":"same"}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        
        # Normalize
        # outputs = normalize(outputs)
        outputs = tf.contrib.layers.layer_norm(outputs)
        outputs = tf.cast(outputs, dtype=tf.float32)
    
    return outputs

def self_attn(enc, scope, 
                dropout, config, reuse):
    for i in range(config.num_blocks):
        with tf.variable_scope(scope+"_encoder_num_blocks_{}".format(i), reuse=reuse):
            ### Multihead Attention
            enc = multihead_attention(queries=enc, 
                                    keys=enc,
                                    scope="multihead_attention_{}".format(i),
                                    num_units=config.hidden_units, 
                                    num_heads=config.num_heads, 
                                    dropout_rate=1 - dropout,
                                    is_training=False,
                                    causality=False,
                                    reuse=reuse)
            
            ### Feed Forward
            enc = feedforward(enc, 
                            num_units=[config.hidden_units, 
                                        config.hidden_units],
                            scope="ffn_{}".format(i),
                            reuse=reuse)

    return enc

