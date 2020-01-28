
# -*- coding: utf-8 -*-
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer

try:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface
except:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface

try:
	from .electra_export import classifier_model_fn_builder
except:
	from electra_export import classifier_model_fn_builder

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

import time, os, sys

def export_api():
	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = model_config_parser(FLAGS)
		
		train_size = int(FLAGS.train_size)
		init_lr = FLAGS.init_lr

		distillation_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))

		model_io_config = Bunch({"fix_lm":False})
		model_io_fn = model_io.ModelIO(model_io_config)
		
		num_classes = FLAGS.num_classes

		model_config_dict = {}
		num_labels_dict = {}
		init_checkpoint_dict = {}
		load_pretrained_dict = {}
		exclude_scope_dict = {}
		not_storage_params_dict = {}
		target_dict = {}

		for task_type in FLAGS.multi_task_type.split(","):
			print("==task type==", task_type)
			model_config_dict[task_type] = model_config_parser(Bunch(distillation_config[task_type]))
			print(task_type, distillation_config[task_type], '=====task model config======')
			num_labels_dict[task_type] = distillation_config[task_type]["num_labels"]
			init_checkpoint_dict[task_type] = os.path.join(FLAGS.buckets, distillation_config[task_type]["init_checkpoint"])
			load_pretrained_dict[task_type] = "yes"
			exclude_scope_dict[task_type] = distillation_config[task_type]["exclude_scope"]
			not_storage_params_dict[task_type] = distillation_config[task_type]["not_storage_params"]
			target_dict[task_type] = distillation_config[task_type]["target"]

		def serving_input_receiver_fn():
			receiver_features = {
				
			}
			print(receiver_features, "==input receiver_features==")
			input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
			return input_fn

		model_fn = model_fn_builder(model_config_dict,
					num_labels_dict,
					init_checkpoint_dict,
					load_pretrained_dict,
					model_io_config=model_io_config,
					opt_config=opt_config,
					exclude_scope_dict=exclude_scope_dict,
					not_storage_params_dict=not_storage_params_dict,
					target_dict=target_dict,
					use_tpu=FLAGS.use_tpu,
					**kargs)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=checkpoint_dir)

	export_dir = estimator.export_savedmodel(export_dir, 
									serving_input_receiver_fn,
									checkpoint_path=init_checkpoint)
	print("===Succeeded in exporting saved model==={}".format(export_dir))
