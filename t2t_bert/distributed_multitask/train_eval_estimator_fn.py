# -*- coding: utf-8 -*-
import tensorflow as tf
from collections import OrderedDict

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils

try:
	from .model_data_interface import data_interface
	from distributed_single_sentence_classification.model_interface import model_config_parser
except:
	from model_data_interface import data_interface
	from distributed_single_sentence_classification.model_interface import model_config_parser

try:
	from .multitask_model_fn import multitask_model_fn
except:
	from multitask_model_fn import multitask_model_fn

# from dataset_generator.input_fn import train_eval_input_fn 

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json, os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
	import paisoar as pai
except Exception as e:
	pai = None

try:
	import horovod.tensorflow as hvd
except Exception as e:
	hvd = None

try:
	import _pickle as pkl
except Exception as e:
	pkl = None

import time

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

		# config = model_config_parser(FLAGS)

		print(FLAGS.train_size)
		
		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / worker_count)
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/worker_count)
			epoch = FLAGS.epoch
		else:
			train_size = int(FLAGS.train_size/worker_count)
			epoch = FLAGS.epoch

		multi_task_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		num_warmup_steps = int(num_train_steps * 0.1)

		num_storage_steps = int(train_size / FLAGS.batch_size)

		num_eval_steps = int(FLAGS.eval_size / FLAGS.batch_size)

		if is_debug == "0":
			num_storage_steps = 190
			num_eval_steps = 100
			num_train_steps = 200
		print("num_train_steps {}, num_eval_steps {}, num_storage_steps {}".format(num_train_steps, num_eval_steps, num_storage_steps))

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============")

		print("==init lr==", FLAGS.init_lr)
		
		opt_config = Bunch({"init_lr":FLAGS.init_lr, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"worker_count":worker_count,
							"opt_type":FLAGS.opt_type,
							"is_chief":is_chief,
							"train_op":kargs.get("train_op", "adam"),
							"decay":kargs.get("decay", "no"),
							"warmup":kargs.get("warmup", "no"),
							"grad_clip":kargs.get("grad_clip", "global_norm"),
							"clip_norm":kargs.get("clip_norm", 1.0),
							"opt_ema":kargs.get("opt_ema", "no")})

		anneal_config = Bunch({
					"initial_value":1.0,
					"num_train_steps":num_train_steps
			})

		model_io_config = Bunch({
								"fix_lm":False,
								"ema_saver":kargs.get("opt_ema", "no")
								})

		if FLAGS.opt_type == "hvd" and hvd:
			checkpoint_dir = checkpoint_dir if task_index == 0 else None
		else:
			checkpoint_dir = checkpoint_dir
		print("==checkpoint_dir==", checkpoint_dir, is_chief)

		model_config_dict = {}
		num_labels_dict = {}
		init_checkpoint_dict = {}
		load_pretrained_dict = {}
		exclude_scope_dict = {}
		not_storage_params_dict = {}
		target_dict = {}
		task_type_dict = {}
		model_type_lst = []
		label_dict = {}

		for task_type in FLAGS.multi_task_type.split(","):
			print("==task type==", task_type)
			multi_task_config[task_type]['buckets'] = FLAGS.buckets
			multi_task_config[task_type]['w2v_path'] = FLAGS.w2v_path
			model_config_dict[task_type] = model_config_parser(Bunch(multi_task_config[task_type]))
			num_labels_dict[task_type] = multi_task_config[task_type]["num_labels"]
			init_checkpoint_dict[task_type] = os.path.join(FLAGS.buckets, multi_task_config[task_type]["init_checkpoint"])
			load_pretrained_dict[task_type] = multi_task_config[task_type]["load_pretrained"]
			exclude_scope_dict[task_type] = multi_task_config[task_type]["exclude_scope"]
			not_storage_params_dict[task_type] = multi_task_config[task_type]["not_storage_params"]
			target_dict[task_type] = multi_task_config[task_type]["target"]
			task_type_dict[task_type] = multi_task_config[task_type]["task_type"]
			label_dict[task_type] = json.load(tf.gfile.Open(os.path.join(FLAGS.buckets,
												multi_task_config[task_type]["label_id"])))

		model_fn = multitask_model_fn(model_config_dict, num_labels_dict,
											task_type_dict,
											init_checkpoint_dict,
											load_pretrained_dict=load_pretrained_dict,
											opt_config=opt_config,
											model_io_config=model_io_config,
											exclude_scope_dict=exclude_scope_dict,
											not_storage_params_dict=not_storage_params_dict,
											target_dict=target_dict,
											output_type="estimator",
											checkpoint_dir=checkpoint_dir,
											num_storage_steps=num_storage_steps,
											anneal_config=anneal_config,
											task_layer_reuse=None,
											model_type_lst=model_type_lst,
											task_invariant=FLAGS.task_invariant,
											multi_task_config=multi_task_config,
											**kargs)

		print("==succeeded in building model==")
		
		name_to_features = data_interface_dual_encoder(FLAGS, multi_task_config, FLAGS.multi_task_type.split(","))

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
			return example

		params = Bunch({})
		params.epoch = epoch
		params.batch_size = FLAGS.batch_size

		if kargs.get("parse_type", "parse_single") == "parse_single":

			train_file_lst = [multi_task_config[task_type]["train_result_file"] for task_type in FLAGS.multi_task_type.split(",")]

			print(train_file_lst)

			train_features = lambda: tf_data_utils.all_reduce_multitask_train_input_fn(train_file_lst,
										_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
										worker_count=worker_count,
										task_index=task_index)

		elif kargs.get("parse_type", "parse_single") == "parse_batch":

			train_file_lst = [multi_task_config[task_type]["train_result_file"] for task_type in FLAGS.multi_task_type.split(",")]
			train_file_path_lst = [os.path.join(FLAGS.buckets, train_file) for train_file in train_file_lst]

			print(train_file_path_lst)
			train_file_path_lst = list(set(train_file_path_lst))

			train_features = lambda: tf_data_utils.all_reduce_train_batch_input_fn(train_file_path_lst,
										_decode_batch_record, 
										name_to_features, 
										params, 
										if_shard=FLAGS.if_shard,
										worker_count=worker_count,
										task_index=task_index)
		# elif kargs.get("parse_type", "parse_single") == "generator":
		# 	def train_features(): return train_eval_input_fn(FLAGS, multi_task_config, "train", 0)

		print("==succeeded in building data and model==")
		print("start training")

		train_hooks = []

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
						config=run_config)

		print("==finish build model estimator==")

		train_being_time = time.time()
		tf.logging.info("==training distribution_strategy=={}".format(kargs.get("distribution_strategy", "MirroredStrategy")))
		if kargs.get("distribution_strategy", "MirroredStrategy") == "MirroredStrategy":
			print("==apply single machine multi-card training==")
			model_estimator.train(input_fn=train_features,
							max_steps=num_train_steps)

			train_end_time = time.time()
			print("==training time==", train_end_time - train_being_time)
			tf.logging.info("==training time=={}".format(train_end_time - train_being_time))
			
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

		
		