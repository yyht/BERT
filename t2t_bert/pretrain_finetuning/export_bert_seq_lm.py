
# -*- coding: utf-8 -*-
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface
except:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface

try:
	from .classifier_fn_tpu_bert_seq_estimator import classifier_model_fn_builder as model_fn_builder
except:
	from classifier_fn_tpu_bert_seq_estimator import classifier_model_fn_builder as model_fn_builder

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

import time, os, sys

def export_model(FLAGS,
				init_checkpoint,
				checkpoint_dir,
				export_dir,
				**kargs):

	config = model_config_parser(FLAGS)
	opt_config = Bunch({})
	anneal_config = Bunch({})
	model_io_config = Bunch({"fix_lm":False})

	# with tf.gfile.Open(FLAGS.label_id, "r") as frobj:
	# 	label_dict = json.load(frobj)

	num_classes = int(FLAGS.num_classes)

	def get_receiver_features():
		receiver_tensors = {
			"input_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids'),
			"segment_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='segment_ids'),
			"input_mask":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_mask'),
			"input_ori_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ori_ids'),
			"context":tf.placeholder(tf.int32, [None, 1], name='context'),
		}
		return receiver_tensors

	def serving_input_receiver_fn():
		receiver_features = get_receiver_features()
		print(receiver_features, "==input receiver_features==")
		input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
		return input_fn

	
	model_fn = model_fn_builder(config, num_classes, init_checkpoint, 
										model_reuse=None, 
										load_pretrained="yes",
										opt_config=opt_config,
										model_io_config=model_io_config,
										exclude_scope="",
										not_storage_params=[],
										target=kargs.get("input_target", ""),
										output_type="estimator",
										checkpoint_dir=checkpoint_dir,
										num_storage_steps=100,
										task_index=0,
										anneal_config=anneal_config,
										**kargs)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=checkpoint_dir)

	export_dir = estimator.export_savedmodel(export_dir, 
									serving_input_receiver_fn,
									checkpoint_path=init_checkpoint)
	print("===Succeeded in exporting saved model==={}".format(export_dir))
