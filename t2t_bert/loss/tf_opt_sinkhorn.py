from utils.bert.bert_utils import get_shape_list

import tensorflow as tf
import numpy as np
from itertools import permutations

def cost_matrix(x,y, x_weight, y_weight, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    # [batch_size, seq, dim]
    x_shape_list = get_shape_list(x, expected_rank=3)
    y_shape_list = get_shape_list(y, expected_rank=3)
    x_col = tf.expand_dims(x, axis=2)
    y_lin = tf.expand_dims(y, axis=1)
    x_weight = tf.expand_dims(x_weight, axis=2)
    y_weight = tf.expand_dims(y_weight, axis=1)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=-1)
    c_weight = (x_weight * y_weight)
    c = c * c_weight
    c += (1.0-c_weight)*1000.0
    return c, c_weight

def sinkhorn_loss(x, y, x_weight, y_weight, epsilon, 
                numItermax=10, stopThr=1e-9, p=2):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    
    Inputs:
        x,y:  The input sets representing the empirical measures.  Each are a tensor of shape (n,D)
        epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
        n:  The number of support points in the empirical measures
        niter:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
    Outputs:
    
    """
    x_shape_list = get_shape_list(x, expected_rank=3)
    batch_size = x_shape_list[0]
    seq_num = x_shape_list[1]
    hidden_dims = x_shape_list[2]

    # The Sinkhorn algorithm takes as input three variables :
    cost, cost_weight = cost_matrix(x, y, x_weight, y_weight, p=p)  # Wasserstein cost function
    
    # both marginals are fixed with equal weights
    x_weight = tf.cast(x_weight, dtype=tf.float32)
    y_weight = tf.cast(y_weight, dtype=tf.float32)
    
    mu = x_weight / tf.reduce_sum(x_weight, axis=-1, keep_dims=True) # [batch_size, seq]
    nu = y_weight / tf.reduce_sum(y_weight, axis=-1, keep_dims=True) # [batch_size, seq]

    u = tf.zeros_like(mu)
    v = tf.zeros_like(nu)

    # To check if algorithm terminates because of threshold
    cpt = tf.constant(0)
    err = tf.constant(1.0)

    c = lambda cpt, u, v, err: tf.logical_and(cpt < numItermax, err > stopThr)

    # Elementary operations
    def M(cost, u, v, cost_weight=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \\epsilon$"
        return (-cost + tf.expand_dims(u, -1) + tf.expand_dims(v, -2) )/epsilon

    def loop_func(cpt, u, v, err):
        u1 = tf.identity(u)
        u = epsilon * x_weight * (tf.log(mu+1e-8) - tf.reduce_logsumexp(M(cost, u, v, cost_weight), axis=-1)) + u
        v = epsilon * y_weight * (tf.log(nu+1e-8) - tf.reduce_logsumexp(tf.transpose(M(cost, u, v, cost_weight), perm=(0, 2, 1)), axis=-1)) + v
        
        err = tf.reduce_mean(tf.reduce_sum(tf.abs(u - u1), axis=-1))

        cpt = tf.add(cpt, 1)
        return cpt, u, v, err

    _, u, v, _ = tf.while_loop(c, loop_func, loop_vars=[cpt, u, v, err])
    pi = tf.exp(M(cost, u, v, cost_weight=None))
    # Sinkhorn distance
    final_cost = tf.reduce_sum(pi * cost * cost_weight, axis=(-2, -1))
    return final_cost
