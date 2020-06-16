import tensorflow as tf
import numpy as np

from model.textcnn import textcnn
from model.textlstm import textlstm
from model.dan import dan

def textcnn_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = True
	else:
		is_training = False

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)

	# if mode == tf.estimator.ModeKeys.PREDICT:
	# 	input_mask = tf.cast(tf.not_equal(input_ids, kargs.get('[PAD]', 0)), tf.int32)
	# 	input_len = tf.reduce_sum(tf.cast(input_mask, tf.int32), -1)
	# 	max_len = tf.reduce_max(input_len, axis=-1)
	# 	input_ids = input_ids[:, :max_len]

	cnn_type = model_config.get("cnn_type", 'dgcnn')

	model = textcnn.TextCNN(model_config)
	model.build_emebdder(input_ids, input_char_ids, is_training, 
						reuse=reuse, 
						cnn_type=cnn_type,
						**kargs)
	model.build_encoder(input_ids, input_char_ids, is_training, 
						reuse=reuse, 
						cnn_type=cnn_type,
						 **kargs)
	return model

def textlstm_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = True
	else:
		is_training = False

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)

	model = textlstm.TextLSTM(model_config)
	model.build_emebdder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	model.build_encoder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	return model

def dan_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = tf.constant(True)
	else:
		is_training = tf.constant(False)

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)

	model = dan.DAN(model_config)
	model.build_emebdder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	model.build_encoder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	return model

# def nvdm_dan_encoder()