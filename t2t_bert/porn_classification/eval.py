import sys,os
sys.path.append("..")

import numpy as np
import tensorflow as tf
from example import hvd_distributed_classifier as bert_classifier
from bunch import Bunch
from data_generator import tokenization
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
from model_io import model_io
from example import feature_writer, write_to_tfrecords
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import horovod.tensorflow as hvd

from optimizer import hvd_distributed_optimizer as optimizer
from porn_classification import classifier_processor

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## Required parameters
flags.DEFINE_string(
	"eval_data_file", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"output_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", 128,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"dev_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_output", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"gpu_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"epoch", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"num_classes", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"train_size", 1402171,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"batch_size", 32,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_type", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"if_shard", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_bool(
	"lower_case", True,
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	hvd.init()

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = json.load(open(FLAGS.config_file, "r"))
		init_checkpoint = FLAGS.init_checkpoint

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		
		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / hvd.size())
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/hvd.size())
			epoch = FLAGS.epoch

		tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, 
		do_lower_case=FLAGS.lower_case)

		classifier_data_api = classifier_processor.EvaluationProcessor()
		classifier_data_api.get_labels(FLAGS.label_id)

		train_examples = classifier_data_api.get_train_examples(FLAGS.train_file)

		write_to_tfrecords.convert_classifier_examples_to_features(train_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer,
																FLAGS.eval_data_file)

		init_lr = 2e-5

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		num_warmup_steps = int(num_train_steps * 0.1)

		num_storage_steps = int(train_size / FLAGS.batch_size)

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============")
		
		opt_config = Bunch({"init_lr":init_lr/hvd.size(), 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps})

		sess = tf.Session(config=sess_config)

		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)

		optimizer_fn = optimizer.Optimizer(opt_config)
		
		num_classes = FLAGS.num_classes
		
		model_eval_fn = bert_classifier.classifier_model_fn_builder(config, num_classes, init_checkpoint, 
												reuse=False, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config)
		
		def metric_fn(features, logits, loss):
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			accuracy = correct = tf.equal(
				tf.cast(pred_label, tf.int32),
				tf.cast(features["label_ids"], tf.int32)
			)
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
			return {"accuracy":accuracy, "loss":loss, "pred_label":pred_label, "label_ids":features["label_ids"]}
		
		name_to_features = {
				"input_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
		}

		def _decode_record(record, name_to_features):
			"""Decodes a record to a TensorFlow example.
			"""
			example = tf.parse_single_example(record, name_to_features)

			# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
			# So cast all int64 to int32.
			for name in list(example.keys()):
				t = example[name]
				if t.dtype == tf.int64:
					t = tf.to_int32(t)
				example[name] = t
			return example 

		params = Bunch({})
		params.epoch = FLAGS.epoch
		params.batch_size = FLAGS.batch_size

		eval_features = tf_data_utils.eval_input_fn(FLAGS.train_result_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard)
		
		[_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
		result = metric_fn(eval_features, eval_logits, eval_loss)
		
		init_op = tf.group(tf.global_variables_initializer(), 
					tf.local_variables_initializer())
		sess.run(init_op)

		sess.run(hvd.broadcast_global_variables(0))

		print("===horovod rank==={}".format(hvd.rank()))
		
		def eval_fn(result):
			i = 0
			total_accuracy = 0
			label, label_id = [], []
			while True:
				try:
					eval_result = sess.run(result)
					total_accuracy += eval_result["accuracy"]
					label_id.extend(eval_result["label_ids"])
					label.extend(eval_result["pred_label"])
					i += 1
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break
			macro_f1 = f1_score(label_id, label, average="macro")
			micro_f1 = f1_score(label_id, label, average="micro")
			macro_precision = precision_score(label_id, label, average="macro")
			micro_precision = precision_score(label_id, label, average="micro")
			macro_recall = recall_score(label_id, label, average="macro")
			micro_recall = recall_score(label_id, label, average="micro")
			accuracy = accuracy_score(label_id, label)
			print("test accuracy {} macro_f1 score {} micro_f1 {} accuracy {}".format(total_accuracy/ i, 
																					macro_f1,  micro_f1, accuracy))
			return total_accuracy/ i, label_id, label
		
		import time
		import time
		start = time.time()
		
		acc, true_label, pred_label = eval_fn(result)
		end = time.time()
		print("==total time {} numbers of devices {}".format(end - start, hvd.size()))
		if hvd.rank() == 0:
			import _pickle as pkl
			pkl.dump({"true_label":true_label, 
						"pred_label":pred_label}, 
						open(FLAGS.model_output+"/predict.pkl", "wb"))

if __name__ == "__main__":
	tf.app.run()