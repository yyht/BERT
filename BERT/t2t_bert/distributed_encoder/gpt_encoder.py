from model.gpt import gpt
import tensorflow as tf
import numpy as np

def gpt_encoder(model_config, features, labels, 
			mode, target, reuse=tf.AUTO_REUSE):

	input_ids = features["input_ids"]
	past = features.get('past', None)

	model = gpt.GPT(model_config)
	if model_config.get("scope", None):
		scope = model_config['scope']
	else:
		scope = 'model'
		model_config['scope'] = scope
	model.build_model(hparams=model_config, 
						X=input_ids, 
						past=past, 
						scope=scope, 
						reuse=reuse)

	return model

