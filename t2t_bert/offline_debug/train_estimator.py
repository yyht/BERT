# -*- coding: utf-8 -*-
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils

try:
	from .model_fn import model_fn_builder
except:
	from model_fn import model_fn_builder

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
	import paisoar as pai
except Exception as e:
	pai = None

try:
	import horovod.tensorflow as hvd
except Exception as e:
	hvd = None

import time, os, sys

def train_eval_fn(FLAGS,
				worker_count, 
				task_index, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir,
				is_debug,
				**kargs):

	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = json.load(open(FLAGS.config_file, "r"))

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"

		config.model = FLAGS.model_type
		config.init_lr = 3e-4
		config.ln_type = FLAGS.ln_type

		print('==init learning rate==', config.init_lr)
		
		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / worker_count)
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/worker_count)
			epoch = FLAGS.epoch
		else:
			train_size = int(FLAGS.train_size/worker_count)
			epoch = FLAGS.epoch

		init_lr = config.init_lr

		label_dict = json.load(tf.gfile.Open(FLAGS.label_id))

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
		
		opt_config = Bunch({"init_lr":init_lr, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"worker_count":worker_count,
							"opt_type":FLAGS.opt_type,
							"is_chief":is_chief,
							"train_op":kargs.get("train_op", "adam"),
							"decay":kargs.get("decay", "no"),
							"warmup":kargs.get("warmup", "no"),
							"clip_norm":1})

		anneal_config = Bunch({
					"initial_value":1.0,
					"num_train_steps":num_train_steps
			})

		model_io_config = Bunch({"fix_lm":False})
		model_io_fn = model_io.ModelIO(model_io_config)
		
		num_classes = FLAGS.num_classes

		if FLAGS.opt_type == "hvd" and hvd:
			checkpoint_dir = checkpoint_dir if task_index == 0 else None
		elif FLAGS.opt_type == "all_reduce":
			checkpoint_dir = checkpoint_dir
		elif FLAGS.opt_type == "collective_reduce":
			checkpoint_dir = checkpoint_dir if task_index == 0 else None
		elif FLAGS.opt_type == "ps" or FLAGS.opt_type == "ps_sync":
			checkpoint_dir = checkpoint_dir if task_index == 0 else None
		print("==checkpoint_dir==", checkpoint_dir, is_chief)

		model_fn = 	model_fn_builder(config, num_classes, init_checkpoint, 
									model_reuse=None, 
									load_pretrained=FLAGS.load_pretrained,
									model_io_config=model_io_config,
									opt_config=opt_config,
									model_io_fn=model_io_fn,
									exclude_scope="",
									not_storage_params=[],
									target=kargs.get("input_target", ""),
									output_type="estimator",
									checkpoint_dir=checkpoint_dir,
									num_storage_steps=num_storage_steps,
									task_index=task_index,
									anneal_config=anneal_config,
									**kargs)

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

		def _decode_batch_record(record, name_to_features):
			example = tf.parse_example(record, name_to_features)
			# for name in list(example.keys()):
			# 	t = example[name]
			# 	if t.dtype == tf.int64:
			# 		t = tf.to_int32(t)
			# 	example[name] = t

			return example

		params = Bunch({})
		params.epoch = FLAGS.epoch
		params.batch_size = FLAGS.batch_size

		
		train_features = lambda: tf_data_utils.all_reduce_train_batch_input_fn(train_file,
									_decode_batch_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)
		eval_features = lambda: tf_data_utils.all_reduce_eval_batch_input_fn(dev_file,
									_decode_batch_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)

		sess_config = tf.ConfigProto(allow_soft_placement=False,
									log_device_placement=False)
		if FLAGS.opt_type == "ps" or FLAGS.opt_type == "ps_sync":
			print("==no need for hook==")
		elif FLAGS.opt_type == "pai_soar" and pai:
			print("no need for hook")
		elif FLAGS.opt_type == "hvd" and hvd:
			sess_config.gpu_options.allow_growth = True
			sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
			print("==no need fo hook==")
		else:
			print("==no need for hooks==")

		if kargs.get("run_config", None):
			run_config = kargs.get("run_config", None)
			run_config = run_config.replace(save_checkpoints_steps=num_storage_steps)
			print("==run config==", run_config.save_checkpoints_steps)
		else:
			run_config = tf.estimator.RunConfig(model_dir=checkpoint_dir, 
											save_checkpoints_steps=num_storage_steps,
											session_config=sess_config)

		train_hooks = []
		if kargs.get("profiler", "profiler") == "profiler":
			if checkpoint_dir:
				hooks = tf.train.ProfilerHook(
							save_steps=100,
							save_secs=None,
							output_dir=os.path.join(checkpoint_dir, "profiler"),
					)
				train_hooks.append(hooks)
				print("==add profiler hooks==")

		model_estimator = tf.estimator.Estimator(
						model_fn=model_fn,
						model_dir=checkpoint_dir,
						config=run_config)

		train_being_time = time.time()
		tf.logging.info("==training distribution_strategy=={}".format(kargs.get("distribution_strategy", "MirroredStrategy")))
		if kargs.get("distribution_strategy", "MirroredStrategy") == "MirroredStrategy":
			print("==apply single machine multi-card training==")
			# model_estimator.train(input_fn=train_features,
			# 				max_steps=num_train_steps,
			# 				hooks=train_hooks)

			train_spec = tf.estimator.TrainSpec(input_fn=train_features, 
											max_steps=num_train_steps)

			eval_spec = tf.estimator.EvalSpec(input_fn=eval_features, 
											steps=num_eval_steps)
			
			model_estimator.train(input_fn=train_features,
							max_steps=num_train_steps,
							hooks=train_hooks)
			# tf.estimator.train(model_estimator, train_spec)
			# tf.estimator.evaluate(model_estimator, eval_spec)


			train_end_time = time.time()
			print("==training time==", train_end_time - train_being_time)
			tf.logging.info("==training time=={}".format(train_end_time - train_being_time))
			eval_results = model_estimator.evaluate(input_fn=eval_features, steps=num_eval_steps)
			# print(eval_results)
			
		elif kargs.get("distribution_strategy", "MirroredStrategy") in ["ParameterServerStrategy", "CollectiveAllReduceStrategy"]: 
			print("==apply multi-machine machine multi-card training==")
			try:
				print(os.environ['TF_CONFIG'], "==tf_run_config==")
			except:
				print("==not tf config==")
			train_spec = tf.estimator.TrainSpec(input_fn=train_features, 
											max_steps=num_train_steps)

			eval_spec = tf.estimator.EvalSpec(input_fn=eval_features, 
											steps=num_eval_steps)

			tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
			train_end_time = time.time()
			print("==training time==", train_end_time - train_being_time)

		
		