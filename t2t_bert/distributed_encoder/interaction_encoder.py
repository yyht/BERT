from model.match_pyramid import match_pyramid
import tensorflow as tf
from model.textcnn import textcnn
from utils.bert import bert_utils
from model.bert import bert
from model.bert import bert_rule
from model.bert import bert_seq
from model.bert import albert
from model.bert import bert_electra_joint
from model.bert import albert_official_electra_joint
from model.bert import albert_official
from model.textcnn import textcnn
import tensorflow as tf
from utils.vae import vae_utils
import numpy as np

def match_pyramid_encoder(model_config, features, labels, 
			mode, target, reuse=None):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = True
	else:
		is_training = False

	input_ids_a = features["input_ids_a"]
	input_char_ids_a = features.get("input_char_ids_a", None)

	input_ids_b = features["input_ids_b"]
	input_char_ids_b = features.get("input_char_ids_b", None)

	model = match_pyramid.MatchPyramid(model_config)
	[emb_seq_a, enc_seq_a, 
	emb_seq_b, enc_seq_b] = model._semantic_encode(input_ids_a, 
											input_char_ids_a, 
											input_ids_b, 
											input_char_ids_b,
											is_training,
											reuse=reuse)

	match_matrix = model._semantic_interaction(input_ids_a, 
								input_char_ids_a, 
								input_ids_b, 
								input_char_ids_b,
								emb_seq_a, 
								enc_seq_a, 
								emb_seq_b, 
								enc_seq_b,
								is_training,
								reuse=reuse)

	print("==match_matrix shape==", match_matrix.get_shape())

	model._semantic_aggerate(match_matrix, 
							is_training,
							dpool_index=features.get("dpool_index", None),
							reuse=reuse)

	return model

def textcnn_interaction_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = True
	else:
		is_training = False

	if mode == tf.estimator.ModeKeys.TRAIN:
		dropout_prob = 0.2
	else:
		dropout_prob = 0.0

	input_ids_a = features["input_ids_a"]
	input_char_ids_a = features.get("input_char_ids_a", None)

	input_ids_b = features["input_ids_b"]
	input_char_ids_b = features.get("input_char_ids_b", None)

	model = textcnn.TextCNN(model_config)

	print(kargs.get('cnn_type', "None"), '==cnn type==')

	model.build_emebdder(input_ids_a, input_char_ids_a, is_training, reuse=tf.AUTO_REUSE, **kargs)
	model.build_encoder(input_ids_a, input_char_ids_a, is_training, reuse=tf.AUTO_REUSE, **kargs)

	# sent_repres_a = model.sent_repres
	sent_repres_a = model.get_pooled_output()

	with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
		hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
		input_ids_a_repres = model.get_pooled_output()
		# input_ids_a_repres = tf.nn.dropout(model.get_pooled_output(), 1-dropout_prob)
		input_ids_a_repres = tf.layers.dense(
						input_ids_a_repres,
						128,
						use_bias=True,
						activation=tf.tanh,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
		# input_ids_a_repres = tf.layers.dense(
		# 				input_ids_a_repres,
		# 				hidden_size,
		# 				use_bias=None,
		# 				activation=None,
		# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

		# input_ids_a_repres = tf.nn.dropout(input_ids_a_repres, keep_prob=1 - dropout_prob)
		# input_ids_a_repres += model.get_pooled_output()
		# input_ids_a_repres = tf.layers.dense(
		# 				input_ids_a_repres,
		# 				hidden_size,
		# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		# 				activation=tf.tanh)

	model.build_emebdder(input_ids_b, input_char_ids_b, is_training, reuse=tf.AUTO_REUSE, **kargs)
	model.build_encoder(input_ids_b, input_char_ids_b, is_training, reuse=tf.AUTO_REUSE, **kargs)

	# sent_repres_b = model.sent_repres
	sent_repres_b = model.get_pooled_output()

	with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
		hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
		input_ids_b_repres = model.get_pooled_output()
		# input_ids_b_repres = tf.nn.dropout(model.get_pooled_output(), 1-dropout_prob)
		input_ids_b_repres = tf.layers.dense(
						input_ids_b_repres,
						128,
						use_bias=True,
						activation=tf.tanh,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
		

		# input_ids_b_repres = tf.layers.dense(
		# 				input_ids_b_repres,
		# 				hidden_size,
		# 				use_bias=None,
		# 				activation=None,
		# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

		# input_ids_b_repres = tf.nn.dropout(input_ids_b_repres, keep_prob=1 - dropout_prob)
		# input_ids_b_repres += model.get_pooled_output()
		# input_ids_b_repres = tf.layers.dense(
		# 				input_ids_b_repres,
		# 				hidden_size,
		# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		# 				activation=tf.tanh)

	concat_repres = tf.concat([input_ids_a_repres, input_ids_b_repres, 
								tf.abs(input_ids_a_repres-input_ids_b_repres),
								input_ids_a_repres*input_ids_b_repres],
								axis=-1)

	# concat_repres = tf.concat([input_ids_a_repres, input_ids_b_repres, 
	# 							tf.abs(input_ids_a_repres-input_ids_b_repres),
	# 							],
	# 							axis=-1)

	feature = {
		"feature_a":input_ids_a_repres,
		"feature_b":input_ids_b_repres,
		"pooled_feature":concat_repres,
		"sent_repres_a":sent_repres_a,
		"sent_repres_b":sent_repres_b
	}

	model.put_task_output(feature)

	return model

def bert_interaction_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = True
	else:
		is_training = False

	if mode == tf.estimator.ModeKeys.TRAIN:
		dropout_prob = 0.2
	else:
		dropout_prob = 0.0

	input_ids_a = features["input_ids_a"]
	input_ids_b = features["input_ids_b"]

	input_mask_a = tf.cast(tf.not_equal(input_ids_a, kargs.get('[PAD]', 0)), tf.int32)
	input_mask_b = tf.cast(tf.not_equal(input_ids_a, kargs.get('[PAD]', 0)), tf.int32)

	segment_ids_a = tf.zeros_like(input_mask_a)
	segment_ids_b = tf.zeros_like(input_mask_b)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	if kargs.get('use_token_type', True):
		tf.logging.info(" use token type ")
	else:
		tf.logging.info(" not use token type ")

	reuse = tf.AUTO_REUSE

	model = bert.Bert(model_config)
	model.build_embedder(input_ids_a, 
						segment_ids_a,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						use_token_type=kargs.get('use_token_type', True),
						reuse=reuse)
	model.build_encoder(input_ids_a,
						input_mask_a,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'))
	model.build_pooler(reuse=reuse)
	sent_repres_a = model.get_pooled_output()
	with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
		hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
		input_ids_a_repres = model.get_pooled_output()
		# input_ids_a_repres = tf.nn.dropout(model.get_pooled_output(), 1-dropout_prob)
		input_ids_a_repres = tf.layers.dense(
						input_ids_a_repres,
						128,
						use_bias=True,
						activation=tf.tanh,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

	model.build_embedder(input_ids_b, 
						segment_ids_b,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						use_token_type=kargs.get('use_token_type', True),
						reuse=reuse)
	model.build_encoder(input_ids_b,
						input_mask_b,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'))
	model.build_pooler(reuse=reuse)
	sent_repres_b = model.get_pooled_output()
	with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
		hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
		input_ids_b_repres = model.get_pooled_output()
		# input_ids_b_repres = tf.nn.dropout(model.get_pooled_output(), 1-dropout_prob)
		input_ids_b_repres = tf.layers.dense(
						input_ids_b_repres,
						128,
						use_bias=True,
						activation=tf.tanh,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

	concat_repres = tf.concat([input_ids_a_repres, input_ids_b_repres, 
								tf.abs(input_ids_a_repres-input_ids_b_repres),
								input_ids_a_repres*input_ids_b_repres],
								axis=-1)

	feature = {
		"feature_a":input_ids_a_repres,
		"feature_b":input_ids_b_repres,
		"pooled_feature":concat_repres,
		"sent_repres_a":sent_repres_a,
		"sent_repres_b":sent_repres_b
	}

	model.put_task_output(feature)

	return model