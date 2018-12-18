import sys,os
sys.path.append("..")
from model_io import model_io
import numpy as np
import tensorflow as tf
from task_pretrain import classifier_fn
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils

from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
from optimizer import hvd_distributed_optimizer as optimizer

import horovod.tensorflow as hvd

flags = tf.flags

FLAGS = flags.FLAGS

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
	"num_classes", 3,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"train_size", 2556200,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"batch_size", 32,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_predictions_per_seq", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"if_shard", "0",
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	hvd.init()

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

	graph = tf.Graph()
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	with graph.as_default():
		import json
		
		# config = json.load(open("/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json", "r"))
		
		config = json.load(open(FLAGS.config_file, "r"))

		init_checkpoint = FLAGS.init_checkpoint
		print("===init checkoutpoint==={}".format(init_checkpoint))

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"

		init_lr = 1e-5

		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / hvd.size())
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/hvd.size())
			epoch = FLAGS.epoch

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		num_warmup_steps = int(num_train_steps * 0.01)

		num_storage_steps = int(train_size / FLAGS.batch_size)

		print(num_train_steps, num_warmup_steps, "=============")
		
		opt_config = Bunch({"init_lr":init_lr/(hvd.size()), 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps})

		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)

		optimizer_fn = optimizer.Optimizer(opt_config)
		
		num_choice = FLAGS.num_classes
		max_seq_length = FLAGS.max_length
		max_predictions_per_seq = FLAGS.max_predictions_per_seq

		model_fn = classifier_fn.classifier_estimator_fn_builder(config, num_calsses, init_checkpoint, 
												reuse=None, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config)
		
		name_to_features = {
				"input_ids":
					tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask":
					tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids":
					tf.FixedLenFeature([max_seq_length], tf.int64),
				"masked_lm_positions":
					tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
					tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
					tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
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
		params.epoch = epoch
		params.batch_size = FLAGS.batch_size

		train_features = tf_data_utils.train_input_fn(FLAGS.train_file,
									_decode_record, name_to_features, params)
		eval_features = tf_data_utils.eval_input_fn(FLAGS.dev_file,
									_decode_record, name_to_features, params)
		
		# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		# sess.run(init_op)

		model_dir = FLAGS.model_output if hvd.rank() == 0 else None

		estimator = tf.estimator.Estimator(
      		model_fn=model_fn,
      		model_dir=model_dir,
      		config=tf.estimator.RunConfig(session_config=session_config),
      		warm_start_from=init_checkpoint)

		bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

		estimator.train(
			input_fn=train_features,
			steps=num_train_steps,
			hooks=[bcast_hook]
			)

		if hvd.rank() == 0:
			eval_result = estimator.evaluate(
				input_fn=eval_features
			)

if __name__ == "__main__":
	flags.mark_flag_as_required("eval_data_file")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("config_file")
	flags.mark_flag_as_required("init_checkpoint")
	flags.mark_flag_as_required("result_file")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("train_file")
	flags.mark_flag_as_required("dev_file")
	flags.mark_flag_as_required("max_length")
	flags.mark_flag_as_required("model_output")
	flags.mark_flag_as_required("gpu_id")
	tf.app.run()
