import tensorflow as tf
import numpy as np

from model.textcnn import textcnn

def textcnn_encoder(model_config, features, labels, 
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

	model = textcnn.TextCNN(model_config)
	model.build_emebdder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	model.build_encoder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	return model
