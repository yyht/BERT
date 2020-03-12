from __future__ import print_function

def multistep(total_steps, cost, cost1, acc):
#     prev_op = tf.no_op()
#     scalar_loss = 0
    tmp_global_step = tf.train.get_or_create_global_step()
    
    with tf.variable_scope('acc', reuse=tf.AUTO_REUSE):
        switch_acc = tf.get_variable(
                                    "switch_acc",
                                    shape=[],
                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                    trainable=False)
            
    def op_1():
        prev_op = tf.no_op()
        for i in range(total_steps):
        # For each substep, make sure that the forward pass ops are created with
        # control dependencies on the train op of the previous substep. We can't
        # just chain the train ops because the weight read for substep n will end up
        # happening before the weights are updated in substep n-1.
            with tf.control_dependencies([prev_op]):
#                 cost, acc = model(x,y)
                prev_op = get_op(cost)
                print(i)
        with tf.control_dependencies([prev_op]):
            prev_op = tmp_global_step.assign_add(1)
        return prev_op
    
    def op_2():
        prev_op = tf.no_op()
        with tf.control_dependencies([prev_op]):
            prev_op = get_op(cost1)
        with tf.control_dependencies([prev_op]):
            no_op = tmp_global_step.assign_add(2)
        return no_op
    
#     with tf.control_dependencies([tf.less(switch_acc, 0.1)]):
#         prev_op = tf.no_op()
#         for i in range(total_steps):
#         # For each substep, make sure that the forward pass ops are created with
#         # control dependencies on the train op of the previous substep. We can't
#         # just chain the train ops because the weight read for substep n will end up
#         # happening before the weights are updated in substep n-1.
#             with tf.control_dependencies([prev_op]):
# #                 cost, acc = model(x,y)
#                 prev_op = get_op(cost)
#                 print(i)
#         with tf.control_dependencies([prev_op]):
#             prev_op = tmp_global_step.assign_add(1)
            
#     with tf.control_dependencies([tf.greater(switch_acc, 0.1)]):
#         no_prev_op = tf.no_op()
#         with tf.control_dependencies([no_prev_op]):
#             no_prev_op = get_op(cost1)
#         with tf.control_dependencies([no_prev_op]):
#             no_op = tmp_global_step.assign_add(2)
            
#     train_op = tf.group(prev_op, no_prev_op)
                
    train_op = tf.cond(tf.less(switch_acc, 0.5), lambda : op_1(), lambda : op_2())
    with tf.control_dependencies([train_op]):
        train_op = switch_acc.assign(acc)
    
    return train_op


def multistep_v1(total_steps, input_train_op):
    train_ops = []
    for i in range(total_steps):
        train_ops.append(input_train_op)
    return tf.group(*train_ops)

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

def model(x, y):

    # Set model weights
    with tf.variable_scope('param', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(shape=[784,10], name='w')
        b = tf.get_variable(shape=[10], name='b')

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    # Minimize error using cross entropy
    # global_step = tf.train.get_or_create_global_step()
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    cost1 = tf.reduce_mean(y-tf.log(pred), reduction_indices=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=-1), tf.argmax(y, axis=-1)), dtype=tf.float32))
    return cost, acc, cost1


def get_op(cost):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = opt.minimize(cost)
    return optimizer
# Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

total_steps = 1
cost, acc, cost1 = model(x,y)
# prev_op = get_op(cost)
# train_op = multistep(total_steps, cost)

def no_op():
    prev_op = tf.no_op()
    with tf.control_dependencies([prev_op]):
        tmp_global_step = tf.train.get_or_create_global_step()
#         train_op = tf.identity(tmp_global_step)
        train_op = tmp_global_step.assign_add(0)
    return train_op


train_op1 = multistep(total_steps, cost, cost1, acc)
global_step = tf.train.get_or_create_global_step()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
#             step = sess.run([global_step])
#             print(step, i, '==prev==')
            acc = sess.run([train_op1], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            step = sess.run([global_step])
            print(step,acc, i, '==after==')
            # Compute average loss
#             avg_cost += c / total_batch
#         # Display logs per epoch step
#         if (epoch+1) % display_step == 0:
#             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
#             print(step)