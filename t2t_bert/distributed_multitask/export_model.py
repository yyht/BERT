try:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from .model_data_interface import data_interface_server
	from .export_model_fn import model_fn_builder
except:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from .model_data_interface import data_interface_server
	from .export_model_fn import model_fn_builder

import json

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json, os
from utils.bert import bert_utils

def export_model(FLAGS,
				init_checkpoint,
				checkpoint_dir,
				export_dir,
				**kargs):

	config = model_config_parser(FLAGS)
	opt_config = Bunch({})
	anneal_config = Bunch({})
	model_io_config = Bunch({"fix_lm":False})

	def serving_input_receiver_fn():
		receiver_features = data_interface_server(FLAGS)
		print(receiver_features, "==input receiver_features==")
		input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
		return input_fn

	model_fn = model_fn_builder(config, 2, init_checkpoint, 
											model_reuse=None, 
											load_pretrained=FLAGS.load_pretrained,
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


