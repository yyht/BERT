import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

EPSILON = 1e-8

class ESIM(ModelTemplate):
    def __init__(self):
        super(ESIM, self).__init__()

    def build_char_embedding(self, char_token, char_lengths, char_embedding, *args, **kargs):
        reuse = kargs["reuse"]
        if self.config.char_embedding == "lstm":
            char_emb = char_embedding_utils.lstm_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        elif self.config.char_embedding == "conv":
            char_emb = char_embedding_utils.conv_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        return char_emb

    def build_emebdding(self, index, *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        if index == "question":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent1_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent1_char, self.sent1_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
        elif index == "passage":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent2_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent2_char, self.sent2_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
        return word_emb
        
    def build_encoder(self, index, input_lengths, *args, **kargs):
        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
            
            if self.config.rnn == "lstm":
                [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                                self.config.context_lstm_dim, 
                                input_lengths=input_lengths, 
                                scope_name=self.config.scope, 
                                reuse=reuse, 
                                is_training=self.is_training,
                                dropout_rate=dropout_rate, 
                                use_cudnn=self.config.use_cudnn)
            elif self.config.rnn == "slstm":
                word_emb_proj = tf.layers.dense(word_emb, 
                                        self.config.slstm_hidden_size)
                initial_hidden_states = word_emb_proj
                initial_cell_states = tf.identity(initial_hidden_states)
                [new_hidden_states, 
                new_cell_states, 
                dummynode_hidden_states] = slstm_utils.slstm_cell(self.config, 
                                    self.config.scope, 
                                    self.config.slstm_hidden_size, 
                                    input_lengths, 
                                    initial_hidden_states, 
                                    initial_cell_states, 
                                    self.config.slstm_layer_num,
                                    dropout_rate, reuse=reuse)
                sent_repres = new_hidden_states
        return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):
        reuse = kargs["reuse"]
        input_dim = sent1_repres.get_shape()[-1]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        with tf.variable_scope(self.config.scope+"_interaction_module", reuse=reuse):
            
            if self.config.with_self_attention:
                v_1_attn = esim_utils.multihead_attention(sent1_repres,
                            sent1_repres,
                            num_units = None,
                            num_heads = 1,
                            dropout_rate = dropout_rate,
                            is_training = True,
                            causality = False,
                            scope = "self_attention",
                            reuse = None)
                v_2_attn = esim_utils.multihead_attention(sent2_repres,
                            sent2_repres,
                            num_units = None,
                            num_heads = 1,
                            dropout_rate = dropout_rate,
                            is_training = True,
                            causality = False,
                            scope = "self_attention",
                            reuse = True)
            
                sent1_repres = tf.concat([sent1_repres, v_1_attn], axis=-1)
                sent2_repres = tf.concat([sent2_repres, v_2_attn], axis=-1)
            [query_attention_outputs, context_attention_outputs] = esim_utils.query_context_attention(sent1_repres, sent2_repres, sent1_len, sent2_len, 
                                        sent1_mask, sent2_mask, dropout_rate,
                                        self.config.scope, reuse=reuse)
            if self.config.get("with_multi_head_interaction", False):
                sent1_repres *= tf.expand_dims(tf.cast(sent1_mask, tf.float32), 2)
                sent2_repres *= tf.expand_dims(tf.cast(sent2_mask, tf.float32), 2)
                q2c = esim_utils.multihead_attention(sent1_repres,
                            sent2_repres,
                            num_units = None,
                            num_heads = self.config.num_heads,
                            dropout_rate = dropout_rate,
                            is_training = True,
                            causality = False,
                            scope = "multihead_attention",
                            reuse = None)
                c2q = esim_utils.multihead_attention(sent2_repres,
                            sent1_repres,
                            num_units = None,
                            num_heads = self.config.num_heads,
                            dropout_rate = dropout_rate,
                            is_training = True,
                            causality = False,
                            scope = "multihead_attention",
                            reuse = True)
                
                query_attention_outputs = tf.concat([query_attention_outputs, c2q], axis=-1)
                query_attention_outputs *= tf.expand_dims(tf.cast(sent1_mask, tf.float32), 2)
                context_attention_outputs = tf.concat([context_attention_outputs, q2c], axis=-1)
                context_attention_outputs *= tf.expand_dims(tf.cast(sent2_mask, tf.float32), 2)
            
            if self.config.rnn == "lstm":
                [sent1_repres_fw, sent1_repres_bw, sent1_repres] = layer_utils.my_lstm_layer(query_attention_outputs, 
                                    self.config.context_lstm_dim, 
                                    input_lengths=sent1_len, 
                                    scope_name=self.config.scope, 
                                    reuse=None, 
                                    is_training=self.is_training,
                                    dropout_rate=dropout_rate, 
                                    use_cudnn=self.config.use_cudnn)
                [sent2_repres_fw, sent2_repres_bw, sent2_repres] = layer_utils.my_lstm_layer(context_attention_outputs, 
                                    self.config.context_lstm_dim, 
                                    input_lengths=sent2_len, 
                                    scope_name=self.config.scope, 
                                    reuse=True, 
                                    is_training=self.is_training,
                                    dropout_rate=dropout_rate, 
                                    use_cudnn=self.config.use_cudnn)
                match_dim = self.config.context_lstm_dim * 8
            elif self.config.rnn == "slstm":
                sent1_initial_hidden_states = tf.layers.dense(query_attention_outputs, 
                                                    self.config.slstm_hidden_size)
                sent1_initial_cell_states = tf.identity(sent1_initial_hidden_states)
                [new_sent1_hidden_states, 
                new_sent1_cell_states, 
                dummynode_sent1_hidden_states] = slstm_utils.slstm_cell(self.config, 
                                    self.config.scope, 
                                    self.config.slstm_hidden_size, 
                                    sent1_len, 
                                    sent1_initial_hidden_states, 
                                    sent1_initial_cell_states, 
                                    self.config.slstm_layer_num,
                                    dropout_rate, reuse=None)
                sent1_repres = new_sent1_hidden_states
                sent2_initial_hidden_states = tf.layers.dense(context_attention_outputs, 
                                                    self.config.slstm_hidden_size)
                sent2_initial_cell_states = tf.identity(sent2_initial_hidden_states)
                [new_sent2_hidden_states, 
                new_sent2_cell_states, 
                dummynode_sent2_hidden_states] = slstm_utils.slstm_cell(self.config, 
                                    self.config.scope, 
                                    self.config.slstm_hidden_size, 
                                    sent2_len, 
                                    sent2_initial_hidden_states, 
                                    sent2_initial_cell_states, 
                                    self.config.slstm_layer_num,
                                    dropout_rate, reuse=True)
                sent2_repres = new_sent2_hidden_states
                match_dim = self.config.slstm_hidden_size * 4
            v_1_sum = tf.reduce_sum(sent1_repres, 1)
            v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(sent1_len, tf.float32)+EPSILON, -1))
            v_2_sum = tf.reduce_sum(sent2_repres, 1)
            v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(sent2_len, tf.float32)+EPSILON, -1))
            # v_1_max = tf.reduce_max(sent1_repres, 1)
            # v_2_max = tf.reduce_max(sent2_repres, 1)
            mask_q = tf.expand_dims(sent1_mask, -1)
            mask_c = tf.expand_dims(sent2_mask, -1)
            v_1_max = tf.reduce_max(qanet_layers.mask_logits(sent1_repres, mask_q), axis=1)
            v_2_max = tf.reduce_max(qanet_layers.mask_logits(sent2_repres, mask_c), axis=1)
            out1 = tf.concat([v_1_ave, v_1_max], axis=-1)
            out2 = tf.concat([v_2_ave, v_2_max], axis=-1)
            out = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)
            
            return out1, out2, out, match_dim

    def build_predictor(self, matched_repres, *args, **kargs):
        match_dim = kargs["match_dim"]
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim
            w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)
            # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
            logits = tf.matmul(matched_repres, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.nn.dropout(logits, (1 - dropout_rate))
            
            self.logits = tf.matmul(logits, w_1) + b_1
            self.pred_probs = tf.nn.softmax(self.logits)

    def build_loss(self, *args, **kargs):
        if self.config.loss == "softmax_loss":
            self.loss, _ = point_wise_loss.softmax_loss(self.logits, self.gold_label, 
                                    *args, **kargs)
        elif self.config.loss == "sparse_amsoftmax_loss":
            self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_binary_v2":
            self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_multi_v1":
            self.loss, _ = point_wise_loss.focal_loss_multi_v1(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
            
    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):
        self.sent1_encoded = self.build_encoder("question", 
                                        self.sent1_token_len, 
                                        reuse = None)
        self.sent2_encoded = self.build_encoder("passage", 
                                        self.sent2_token_len, 
                                        reuse = True)
        [self.sent1_repres, 
        self.sent2_repres, 
        self.aggregat_repres,
        match_dim] = self.build_interactor(self.sent1_encoded, 
                                    self.sent2_encoded,
                                    self.sent1_token_len, 
                                    self.sent2_token_len,
                                    self.sent1_token_mask,
                                    self.sent2_token_mask,
                                    reuse = None)
        self.build_predictor(self.aggregat_repres,
                            reuse = None,
                            match_dim = match_dim)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent1_token, sent2_token, gold_label] = sample_batch
        feed_dict = {
            self.sent1_token: sent1_token,
            self.sent2_token: sent2_token,
            self.gold_label:gold_label,
            self.learning_rate: self.config.learning_rate,
            self.is_training:kargs["is_training"]
        }
        return feed_dict