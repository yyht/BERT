import sys,os
sys.path.append("..")

import tensorflow as tf
import bert_order_estimator as bert_classifier_estimator
from model_io import model_io
from bunch import Bunch
import 	json

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

def serving_input_receiver_fn(feature_spec, max_seq_length):
	"""An input receiver that expects a serialized tf.Example."""

	serialized_tf_example = tf.placeholder(dtype=tf.string,
									shape=None,
									name='input_example_tensor')
	receiver_tensors = {'examples': serialized_tf_example}
	features = tf.parse_example(serialized_tf_example, feature_spec)
	return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def export_model(config):

	opt_config = Bunch({"init_lr":2e-5, "num_train_steps":1e30, "cycle":False})
	model_io_config = Bunch({"fix_lm":False})

	bert_config = json.load(open(config["config_file"], "r"))
	model_config = Bunch(bert_config)

	with open(config["label2id"], "r") as frobj:
		label_dict = json.load(frobj)

	num_classes = len(label_dict["id2label"])
	max_seq_length = config["max_length"]

	feature_spec = {
				"input_ids_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_ids_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
	}

	model_io_fn = model_io.ModelIO(model_io_config)

	model_fn = bert_classifier_estimator.classifier_model_fn_builder(
									model_config, 
									num_classes, 
									config["init_checkpoint"], 
									model_reuse=None, 
									load_pretrained=True,
									model_io_fn=model_io_fn,
									model_io_config=model_io_config, 
									opt_config=opt_config,
									input_name=["a", "b"],
									label_tensor=None)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=config["model_dir"])

	export_dir = estimator.export_savedmodel(export_dir_base=config["export_path"], 
									serving_input_receiver_fn=serving_input_receiver_fn(feature_spec, max_seq_length))

	print("===Succeeded in exporting saved model===")

if __name__ == "__main__":

	model_config = {
		"label2id":FLAGS.label2id,
		"init_checkpoint":FLAGS.init_checkpoint,
		"config_file":FLAGS.config_file,
		"max_length":FLAGS.max_length,
		"model_dir":FLAGS.model_dir,
		"export_path":FLAGS.export_path
	}

	export_model(model_config)


