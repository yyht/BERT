import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils

from bert_model_fn import model_fn_builder

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
	import tensorflow.contrib.pai_soar as pai
except Exception as e:
	pai = None

try:
	import horovod.tensorflow as hvd
except Exception as e:
	hvd = None

def train_eval_fn(FLAGS,
				worker_count, 
				task_index, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir,
				is_debug):

	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = json.load(open(FLAGS.config_file, "r"))

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		
		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / worker_count)
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/worker_count)
			epoch = FLAGS.epoch

		init_lr = 2e-5

		label_dict = json.load(open(FLAGS.label_id))

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		num_warmup_steps = int(num_train_steps * 0.1)

		num_storage_steps = int(train_size / FLAGS.batch_size)

		num_eval_steps = int(FLAGS.eval_size / FLAGS.batch_size)

		if is_debug == "0":
			num_storage_steps = 2
			num_eval_steps = 10
			num_train_steps = 10
		print("num_train_steps {}, num_eval_steps {}, num_storage_steps {}".format(num_train_steps, num_eval_steps, num_storage_steps))

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============")
		
		opt_config = Bunch({"init_lr":init_lr/worker_count, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"worker_count":worker_count,
							"opt_type":FLAGS.opt_type})

		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)

		optimizer_fn = optimizer.Optimizer(opt_config)
		
		num_classes = FLAGS.num_classes

		model_fn = model_fn_builder(config, num_classes, init_checkpoint, 
												model_reuse=None, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config,
												exclude_scope="",
												not_storage_params=[],
												target="")

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

		train_features = lambda: tf_data_utils.train_input_fn(train_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)

		eval_features = lambda: tf_data_utils.eval_input_fn(dev_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)

		print("===========begin to train============")
		sess_config = tf.ConfigProto(allow_soft_placement=False,
									log_device_placement=False)

		checkpoint_dir = checkpoint_dir if task_index == 0 else None

		print("start training")

		hooks = []
		if FLAGS.opt_type == "ps":
			sync_replicas_hook = optimizer_fn.opt.make_session_run_hook(is_chief, num_tokens=0)
			hooks.append(sync_replicas_hook)
			
		elif FLAGS.opt_type == "pai_soar" and pai:
			sess = tf.train.MonitoredTrainingSession(master=target,
												 is_chief=is_chief,
												 config=sess_config,
												 hooks=hooks,
												 checkpoint_dir=checkpoint_dir,
												 save_checkpoint_steps=num_storage_steps)
		elif FLAGS.opt_type == "hvd" and hvd:
			bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
			hooks.append(bcast_hook)
			sess_config.gpu_options.allow_growth = True
			sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
		else:
			print("==not supported==")

		run_config = tf.estimator.RunConfig(model_dir=checkpoint_dir, 
											save_checkpoints_steps=num_storage_steps,
											session_config=sess_config)
		
		estimator = tf.estimator.Estimator(
				        model_fn=model_fn,
				        config=run_config)

		train_spec = tf.estimator.TrainSpec(input_fn=train_features, max_steps=num_train_steps)
		eval_spec = tf.estimator.EvalSpec(input_fn=eval_features, steps=num_eval_steps)

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
		