import tensorflow as tf
import numpy as np
from collections import Counter
from bunch import Bunch
import os, sys

from model_io import model_io
from optimizer import distributed_optimizer as optimizer
from utils.bert import bert_utils
try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo


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
					label_lst=None,
					output_type="sess",
					**kargs):

	def model_fn(features, labels, mode):

		model_api = model_zoo(model_config)
		model = model_api(model_config, features, labels,
							mode, target, reuse=model_reuse, **kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = 0.2
		else:
			dropout_prob = 0.0

		with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
			hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
			feature_output_a = tf.layers.dense(
							model.get_pooled_output(),
							hidden_size,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
			feature_output_a = tf.nn.dropout(feature_output_a, keep_prob=1 - dropout_prob)
			feature_output_a += model.get_pooled_output()
			sentence_pres = tf.layers.dense(
							feature_output_a,
							hidden_size,
							kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
							activation=tf.tanh)
		sentence_pres = tf.nn.l2_normalize(sentence_pres, axis=-1)

		model_io_fn = model_io.ModelIO(model_io_config)

		tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		try:
			params_size = model_io_fn.count_params(model_config.scope)
			print("==total params==", params_size)
		except:
			print("==not count params==")
		print(tvars)
		if load_pretrained == "yes":
			model_io_fn.load_pretrained(tvars, 
										init_checkpoint,
										exclude_scope=exclude_scope)

		estimator_spec = tf.estimator.EstimatorSpec(
										mode=mode,
										predictions={
													'sentence_pres':sentence_pres
										},
										export_outputs={
											"output":tf.estimator.export.PredictOutput(
														{
															'sentence_pres':sentence_pres
														}
													)
										}
							)
		return estimator_spec
	return model_fn