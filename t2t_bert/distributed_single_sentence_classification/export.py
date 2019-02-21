import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils

try:
	from .bert_model_fn import model_fn_builder
	from .bert_model_fn import rule_model_fn_builder
except:
	from bert_model_fn import model_fn_builder
	from bert_model_fn import rule_model_fn_builder

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

def export_model_v1(config,
					**kargs):

	bert_config = json.load(open(config["config_file"], "r"))
	model_config = Bunch(bert_config)

	model_config.use_one_hot_embeddings = True
	model_config.scope = "bert"
	model_config.dropout_prob = 0.1
	model_config.label_type = "single_label"

	with open(config["label2id"], "r") as frobj:
		label_dict = json.load(frobj)

	num_classes = len(label_dict["id2label"])
	max_seq_length = config["max_length"]

	def serving_input_receiver_fn():
		# receive tensors
		receiver_tensors = {
				"input_ids":
						tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids'),
				"input_mask":
						tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask'),
				"segment_ids":
						tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids'),
				"label_ids":
						tf.placeholder(tf.int32, [None], name='label_ids'),
		}

		# Convert give inputs to adjust to the model.
		features = {}
		for key in receiver_tensors:
			features[key] = receiver_tensors[key]
		return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
													features=features)

	# def serving_input_receiver_fn():
	# 	receive serialized example
	# 	serialized_tf_example = tf.placeholder(dtype=tf.string,
	# 									shape=None,
	# 									name='input_example_tensor')
	# 	receiver_tensors = {'examples': serialized_tf_example}
	# 	features = tf.parse_example(serialized_tf_example, feature_spec)
	# 	return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

	if config.get("rule_model", "rule"):
		model_fn_interface = rule_model_fn_builder
	else:
		model_fn_interface = model_fn_builder

	opt_config = Bunch({"train_op":"adam"})
	model_io_config = Bunch({"fix_lm":False})

	model_fn = model_fn_interface(config, num_classes, init_checkpoint, 
								model_reuse=None, 
								load_pretrained=True,
								opt_config=opt_config,
								model_io_config=model_io_config,
								exclude_scope="",
								not_storage_params=[],
								target="",
								output_type="estimator",
								checkpoint_dir=self.config["model_dir"],
								num_storage_steps=1000,
								task_index=0)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=config["model_dir"])

	export_dir = estimator.export_savedmodel(config["export_path"], 
									serving_input_receiver_fn,
									checkpoint_path=config["init_checkpoint"])

	print("===Succeeded in exporting saved model==={}".format(export_dir))

def export_model_v2(config):

	bert_config = json.load(open(config["config_file"], "r"))
	model_config = Bunch(bert_config)

	model_config.use_one_hot_embeddings = True
	model_config.scope = "bert"
	model_config.dropout_prob = 0.1
	model_config.label_type = "single_label"

	with open(config["label2id"], "r") as frobj:
		label_dict = json.load(frobj)

	num_classes = len(label_dict["id2label"])
	max_seq_length = config["max_length"]

	def serving_input_receiver_fn():
		label_ids = tf.placeholder(tf.int32, [None], name='label_ids')

		input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
		input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
		segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')

		input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
			'label_ids': label_ids,
			'input_ids': input_ids,
			'input_mask': input_mask,
			'segment_ids': segment_ids
		})()
		return input_fn

	if config.get("rule_model", "rule"):
		model_fn_interface = rule_model_fn_builder
	else:
		model_fn_interface = model_fn_builder

	opt_config = Bunch({"train_op":"adam"})
	model_io_config = Bunch({"fix_lm":False})

	model_fn = model_fn_interface(config, num_classes, config["init_checkpoint"], 
								model_reuse=None, 
								load_pretrained=True,
								opt_config=opt_config,
								model_io_config=model_io_config,
								exclude_scope="",
								not_storage_params=[],
								target="",
								output_type="estimator",
								checkpoint_dir=config["model_dir"],
								num_storage_steps=1000,
								task_index=0)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=config["model_dir"])

	export_dir = estimator.export_savedmodel(config["export_path"], 
									serving_input_receiver_fn,
									checkpoint_path=config["init_checkpoint"])
	print("===Succeeded in exporting saved model==={}".format(export_dir))


