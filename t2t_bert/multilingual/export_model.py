import sys,os
sys.path.append("..")

import tensorflow as tf
import bert_classifier_estimator
from model_io import model_io
from bunch import Bunch
import 	json

from optimizer import hvd_distributed_optimizer as optimizer

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_dir", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label2id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"export_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"export_type", None,
	"Input TF example files (can be a glob or comma separated).")

def export_model_v1(config):

	opt_config = Bunch({"init_lr":2e-5, "num_train_steps":1e30, "cycle":False})
	model_io_config = Bunch({"fix_lm":False})

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

	model_io_fn = model_io.ModelIO(model_io_config)

	model_fn = bert_classifier_estimator.classifier_model_fn_builder(
									model_config, 
									num_classes, 
									config["init_checkpoint"], 
									reuse=None, 
									load_pretrained=True,
									model_io_fn=model_io_fn,
									model_io_config=model_io_config, 
									opt_config=opt_config)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=config["model_dir"])

	export_dir = estimator.export_savedmodel(config["export_path"], 
									serving_input_receiver_fn,
									checkpoint_path=config["init_checkpoint"])

	print("===Succeeded in exporting saved model==={}".format(export_dir))

def export_model_v2(config):

	opt_config = Bunch({"init_lr":2e-5, "num_train_steps":1e30, "cycle":False})
	model_io_config = Bunch({"fix_lm":False})

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

	opt_config = Bunch({"init_lr":0.1, 
							"num_train_steps":10,
							"num_warmup_steps":10})

	optimizer_fn = optimizer.Optimizer(opt_config)

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

	model_io_fn = model_io.ModelIO(model_io_config)

	model_fn = bert_classifier_estimator.classifier_model_fn_builder(
									model_config, 
									num_classes, 
									config["init_checkpoint"], 
									reuse=None, 
									load_pretrained=True,
									model_io_fn=model_io_fn,
									model_io_config=model_io_config, 
									opt_config=opt_config)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=config["model_dir"])

	export_dir = estimator.export_savedmodel(config["export_path"], 
									serving_input_receiver_fn,
									checkpoint_path=config["init_checkpoint"])

	print("===Succeeded in exporting saved model==={}".format(export_dir))

if __name__ == "__main__":

	model_config = {
		"label2id":FLAGS.label2id,
		"init_checkpoint":FLAGS.init_checkpoint,
		"config_file":FLAGS.config_file,
		"max_length":FLAGS.max_length,
		"model_dir":FLAGS.model_dir,
		"export_path":FLAGS.export_path
	}
	if FLAGS.export_type == "1":
		export_model_v1(model_config)
	elif FLAGS.export_type == "2":
		export_model_v2(model_config)


