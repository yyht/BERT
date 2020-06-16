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
from utils.simclr import simclr_utils


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
							tf.estimator.ModeKeys.PREDICT, 
							target, reuse=model_reuse,
							**kargs)

		dropout_prob = 0.0
		is_training = False

		with tf.variable_scope(model_config.scope+"/feature_output", reuse=tf.AUTO_REUSE):
			hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
			sentence_pres = model.get_pooled_output()

			sentence_pres = tf.layers.dense(
						sentence_pres,
						128,
						use_bias=True,
						activation=tf.tanh,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

			# sentence_pres = tf.layers.dense(
			# 				model.get_pooled_output(),
			# 				hidden_size,
			# 				use_bias=None,
			# 				activation=tf.nn.relu,
			# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
			
			# sentence_pres = tf.layers.dense(
			# 				sentence_pres,
			# 				hidden_size,
			# 				use_bias=None,
			# 				activation=None,
			# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

			# hidden_size = bert_utils.get_shape_list(model.get_pooled_output(), expected_rank=2)[-1]
			# sentence_pres = tf.layers.dense(
			# 			model.get_pooled_output(),
			# 			hidden_size,
			# 			use_bias=True,
			# 			activation=tf.tanh,
			# 			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
			# feature_output_a = tf.layers.dense(
			# 				model.get_pooled_output(),
			# 				hidden_size,
			# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
			# feature_output_a = tf.nn.dropout(feature_output_a, keep_prob=1 - dropout_prob)
			# feature_output_a += model.get_pooled_output()
			# sentence_pres = tf.layers.dense(
			# 				feature_output_a,
			# 				hidden_size,
			# 				kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			# 				activation=tf.tanh)

		if kargs.get('apply_head_proj', False):
			with tf.variable_scope(model_config.scope+"/head_proj", reuse=tf.AUTO_REUSE):
				sentence_pres = simclr_utils.projection_head(sentence_pres, 
										is_training, 
										head_proj_dim=128,
										num_nlh_layers=1,
										head_proj_mode='nonlinear',
										name='head_contrastive')

		l2_sentence_pres = tf.nn.l2_normalize(sentence_pres+1e-20, axis=-1)

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
										mode=tf.estimator.ModeKeys.PREDICT,
										predictions={
													'sentence_pres':l2_sentence_pres,
													# "before_l2":sentence_pres
										},
										export_outputs={
											"output":tf.estimator.export.PredictOutput(
														{
															'sentence_pres':l2_sentence_pres,
															# "before_l2":sentence_pres
														}
													)
										}
							)
		return estimator_spec
	return model_fn