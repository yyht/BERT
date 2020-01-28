import tensorflow as tf
import numpy as np
from collections import Counter

from model_io import model_io
from optimizer import distributed_optimizer as optimizer

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

from task_module import classifier

def model_fn_builder(
					model_config,
					num_labels,
					init_checkpoint,
					model_reuse=None,
					load_pretrained=True,
					model_io_config={},
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="a",
					**kargs):

	def model_fn(features, labels, mode):

		model_api = model_zoo(model_config)

		model = model_api(model_config, features, labels,
							mode, target, reuse=model_reuse)

		label_ids = features["label_ids"]

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope

		with tf.variable_scope(scope):
			(loss, 
			per_example_loss, 
			logits) = classifier.classifier(model_config,
											model.get_pooled_output(),
											num_labels,
											label_ids,
											dropout_prob)

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		try:
			params_size = model_io_fn.count_params(model_config.scope)
			print("==total params==", params_size)
		except:
			print("==not count params==")

		if load_pretrained == "yes":
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)
		return_dict = {
					"loss":loss, 
					"logits":logits,
					"tvars":tvars,
					"model":model,
					"per_example_loss":per_example_loss
				}
		return return_dict
	return model_fn
		

		