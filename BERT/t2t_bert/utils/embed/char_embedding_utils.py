import tensorflow as tf
from utils.bimpm import layer_utils, match_utils
from utils.qanet import qanet_layers

def lstm_char_embedding(char_token, char_lengths, char_embedding, 
    config, is_training=True, reuse=None):
    dropout_rate = tf.cond(is_training, 
                        lambda:config.dropout_rate, 
                        lambda:0.0)

    with tf.variable_scope(config.scope+"_lstm_char_embedding_layer", reuse=reuse):
        char_dim = char_embedding.get_shape()[-1]
        input_shape = tf.shape(char_token)
        batch_size = input_shape[0]
        question_len = input_shape[1]
        char_len = input_shape[2]

        in_question_char_repres = tf.nn.embedding_lookup(char_embedding, char_token)
        in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, char_len, char_dim])
        question_char_lengths = tf.reshape(char_lengths, [-1])
        quesiton_char_mask = tf.sequence_mask(question_char_lengths, char_len, dtype=tf.float32)  # [batch_size*question_len, q_char_len]
        in_question_char_repres = tf.multiply(in_question_char_repres, tf.expand_dims(quesiton_char_mask, axis=-1))

        (question_char_outputs_fw, question_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_question_char_repres, config.char_lstm_dim,
                input_lengths=question_char_lengths,scope_name="char_lstm", reuse=reuse,
                is_training=is_training, dropout_rate=dropout_rate, use_cudnn=config.use_cudnn)
        question_char_outputs_fw = layer_utils.collect_final_step_of_lstm(question_char_outputs_fw, question_char_lengths - 1)
        question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
        question_char_outputs = tf.concat(axis=1, values=[question_char_outputs_fw, question_char_outputs_bw])
        question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, 2*config.char_lstm_dim])

        return question_char_outputs

def conv_char_embedding(char_token, char_lengths, char_embedding, 
    config, is_training=True, reuse=None):
    dropout_rate = tf.cond(is_training, 
                    lambda:config.dropout_rate, 
                    lambda:0.0)

    with tf.variable_scope(config.scope+"_conv_char_embedding_layer", reuse=reuse):
        char_dim = char_embedding.get_shape()[-1]
        input_shape = tf.shape(char_token)
        batch_size = input_shape[0]
        question_len = input_shape[1]
        char_len = input_shape[2]
        ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    char_embedding, char_token), [-1, char_len, char_dim])

        ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * dropout_rate)
        ch_emb = conv(ch_emb, config.char_conv_dim,
                    bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = reuse)
        ch_emb = tf.reduce_max(ch_emb, axis = 1)
        ch_emb = tf.reshape(ch_emb, [-1, question_len, ch_emb.shape[-1]])
        return ch_emb

