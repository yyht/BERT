import tensorflow as tf
import numpy as np

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

def get_hidden_states_before(hidden_states, step, shape, hidden_size):
    #padding zeros
    padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
    #remove last steps
    displaced_hidden_states=hidden_states[:,:-step,:]
    #concat padding
    return tf.concat([padding, displaced_hidden_states], axis=1)
    #return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1), lambda: tf.zeros((shape[0], shape[1], config.hidden_size_sum), dtype=tf.float32))

def get_hidden_states_after(hidden_states, step, shape, hidden_size):
    #padding zeros
    padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
    #remove last steps
    displaced_hidden_states=hidden_states[:,step:,:]
    #concat padding
    return tf.concat([displaced_hidden_states, padding], axis=1)
    #return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1), lambda: tf.zeros((shape[0], shape[1], config.hidden_size_sum), dtype=tf.float32))

def sum_together(l):
    combined_state=None
    for tensor in l:
        if combined_state==None:
            combined_state=tensor
        else:
            combined_state=combined_state+tensor
    return combined_state
    
def slstm_cell(config, scope, hidden_size, 
            lengths, initial_hidden_states, initial_cell_states, num_layers,
            dropout, reuse=None):
    with tf.variable_scope(scope, reuse = reuse):
        #Word parameters 
        #forget gate for left 
        with tf.variable_scope("f1_gate", reuse = reuse):
            #current
            Wxf1 = tf.get_variable("Wxf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            #left right
            Whf1 = tf.get_variable("Whf", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            #initial state
            Wif1 = tf.get_variable("Wif1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            #dummy node
            Wdf1 = tf.get_variable("Wdf1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
       
        #forget gate for right 
        with tf.variable_scope("f2_gate", reuse = reuse):

            Wxf2 = tf.get_variable("Wxf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Whf2 = tf.get_variable("Whf", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            Wif2 = tf.get_variable("Wif1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Wdf2 = tf.get_variable("Wdf1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)

        #forget gate for inital states   
        with tf.variable_scope("f3_gate", reuse = reuse):

            Wxf3 = tf.get_variable("Wxf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Whf3 = tf.get_variable("Whf", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            Wif3 = tf.get_variable("Wif1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Wdf3 = tf.get_variable("Wdf1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)  
        #forget gate for dummy states 
        with tf.variable_scope("f4_gate", reuse = reuse):

            Wxf4 = tf.get_variable("Wxf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Whf4 = tf.get_variable("Whf", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            Wif4 = tf.get_variable("Wif1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Wdf4 = tf.get_variable("Wdf1", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
        #input gate for current state     
        with tf.variable_scope("i_gate", reuse = reuse):

            Wxi = tf.get_variable("Wxi", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Whi = tf.get_variable("Whi", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            Wii = tf.get_variable("Wii", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Wdi = tf.get_variable("Wdi", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)  
        
        #input gate for output gate
        with tf.variable_scope("o_gate", reuse = reuse):

            Wxo = tf.get_variable("Wxo", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Who = tf.get_variable("Who", dtype=tf.float32,
                                    shape=[2*hidden_size, hidden_size],
                                    initializer=initializer)
            Wio = tf.get_variable("Wio", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            Wdo = tf.get_variable("Wdo", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)

        with tf.variable_scope("biases", reuse = reuse):
            bi = tf.get_variable("bi", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            bo = tf.get_variable("bo", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            bf1 = tf.get_variable("bf1", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            bf2 = tf.get_variable("bf2", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            bf3 = tf.get_variable("bf3", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            bf4 = tf.get_variable("bf4", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)

        #dummy node gated attention parameters
        #input gate for dummy state
        with tf.variable_scope("gated_d_gate", reuse = reuse):
            gated_Wxd = tf.get_variable("gated_Wxd", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            gated_Whd = tf.get_variable("gated_Whd", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
        #output gate
        with tf.variable_scope("gated_o_gate", reuse = reuse):
            gated_Wxo = tf.get_variable("gated_Wxo", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            gated_Who = tf.get_variable("gated_Who", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
        #forget gate for states of word
        with tf.variable_scope("gated_f_gate", reuse = reuse):
            gated_Wxf = tf.get_variable("gated_Wxf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
            gated_Whf = tf.get_variable("gated_Whf", dtype=tf.float32,
                                    shape=[hidden_size, hidden_size],
                                    initializer=initializer)
        #biases
        with tf.variable_scope("gated_biases", reuse = reuse):
            gated_bd = tf.get_variable("gated_bd", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            gated_bo = tf.get_variable("gated_bo", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)
            gated_bf = tf.get_variable("gated_bf", dtype=tf.float32,
                                    shape=[hidden_size],
                                    initializer=initializer)

    #filters for attention        
    mask_softmax_score=tf.cast(tf.sequence_mask(lengths), tf.float32)*1e25-1e25
    mask_softmax_score_expanded=tf.expand_dims(mask_softmax_score, dim=2)               
    #filter invalid steps
    sequence_mask=tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32),axis=2)
    #filter embedding states
    initial_hidden_states=initial_hidden_states*sequence_mask
    initial_cell_states=initial_cell_states*sequence_mask
    #record shape of the batch
    shape=tf.shape(initial_hidden_states)
    
    #initial embedding states
    embedding_hidden_state=tf.reshape(initial_hidden_states, [-1, hidden_size])      
    embedding_cell_state=tf.reshape(initial_cell_states, [-1, hidden_size])

    #randomly initialize the states
    if config.random_initialize:
        initial_hidden_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
        initial_cell_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
        #filter it
    initial_hidden_states=initial_hidden_states*sequence_mask
    initial_cell_states=initial_cell_states*sequence_mask

    #inital dummy node states
    dummynode_hidden_states=tf.reduce_mean(initial_hidden_states, axis=1)
    dummynode_cell_states=tf.reduce_mean(initial_cell_states, axis=1)

    for i in range(num_layers):
        #update dummy node states
        #average states
        combined_word_hidden_state=tf.reduce_mean(initial_hidden_states, axis=1)
        reshaped_hidden_output=tf.reshape(initial_hidden_states, [-1, hidden_size])
        #copy dummy states for computing forget gate
        transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
        #input gate
        gated_d_t = tf.nn.sigmoid(
            tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd
        )
        #output gate
        gated_o_t = tf.nn.sigmoid(
            tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo
        )
        #forget gate for hidden states
        gated_f_t = tf.nn.sigmoid(
            tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf
        )

        #softmax on each hidden dimension 
        reshaped_gated_f_t=tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size])+ mask_softmax_score_expanded
        gated_softmax_scores=tf.nn.softmax(tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
        #split the softmax scores
        new_reshaped_gated_f_t=gated_softmax_scores[:,:shape[1],:]
        new_gated_d_t=gated_softmax_scores[:,shape[1]:,:]
        #new dummy states
        dummy_c_t=tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) + tf.squeeze(new_gated_d_t, axis=1)*dummynode_cell_states
        dummy_h_t=gated_o_t * tf.nn.tanh(dummy_c_t)

        #update word node states
        #get states before
        initial_hidden_states_before=[tf.reshape(get_hidden_states_before(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(config.step)]
        initial_hidden_states_before=sum_together(initial_hidden_states_before)
        initial_hidden_states_after= [tf.reshape(get_hidden_states_after(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(config.step)]
        initial_hidden_states_after=sum_together(initial_hidden_states_after)
        #get states after
        initial_cell_states_before=[tf.reshape(get_hidden_states_before(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(config.step)]
        initial_cell_states_before=sum_together(initial_cell_states_before)
        initial_cell_states_after=[tf.reshape(get_hidden_states_after(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(config.step)]
        initial_cell_states_after=sum_together(initial_cell_states_after)
        
        #reshape for matmul
        initial_hidden_states=tf.reshape(initial_hidden_states, [-1, hidden_size])
        initial_cell_states=tf.reshape(initial_cell_states, [-1, hidden_size])

        #concat before and after hidden states
        concat_before_after=tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

        #copy dummy node states 
        transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
        transformed_dummynode_cell_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1],1]), [-1, hidden_size])

        f1_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) + 
            tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1)+ bf1
        )

        f2_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) + 
            tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2)+ bf2
        )

        f3_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) + 
            tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
        )

        f4_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) + 
            tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
        )
        
        i_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) + 
            tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi)+ bi
        )
        
        o_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) + 
            tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
        )
        
        f1_t, f2_t, f3_t, f4_t, i_t=tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1),tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)


        five_gates=tf.concat([f1_t, f2_t, f3_t, f4_t,i_t], axis=1)
        five_gates=tf.nn.softmax(five_gates, dim=1)
        f1_t,f2_t,f3_t, f4_t,i_t= tf.split(five_gates, num_or_size_splits=5, axis=1)
        
        f1_t, f2_t, f3_t, f4_t, i_t=tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1),tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1),tf.squeeze(i_t, axis=1)

        c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after)+(f3_t * embedding_cell_state)+ (f4_t * transformed_dummynode_cell_states)+ (i_t * initial_cell_states)
        
        h_t = o_t * tf.nn.tanh(c_t)

        #update states
        initial_hidden_states=tf.reshape(h_t, [shape[0], shape[1], hidden_size])
        initial_cell_states=tf.reshape(c_t, [shape[0], shape[1], hidden_size])
        initial_hidden_states=initial_hidden_states*sequence_mask
        initial_cell_states=initial_cell_states*sequence_mask

        dummynode_hidden_states=dummy_h_t
        dummynode_cell_states=dummy_c_t

    initial_hidden_states = tf.nn.dropout(initial_hidden_states, 1 - dropout)
    initial_cell_states = tf.nn.dropout(initial_cell_states, 1 - dropout)

    return initial_hidden_states, initial_cell_states, dummynode_hidden_states