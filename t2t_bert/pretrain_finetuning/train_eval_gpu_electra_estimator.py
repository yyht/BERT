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
	from .electra_model_fn import classifier_model_fn_builder
except:
	from electra_model_fn import classifier_model_fn_builder

try:
	from .electra_model_fn_gumbel import classifier_model_fn_builder as classifier_model_fn_builder_gumbel
except:
	from electra_model_fn_gumbel import classifier_model_fn_builder as classifier_model_fn_builder_gumbel

import numpy as np
import tensorflow as tf
from bunch import Bunch
from model_io import model_io
import json

import time, os, sys

def train_eval_fn(FLAGS,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir,
				**kargs):

	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = model_config_parser(FLAGS)
		
		train_size = int(FLAGS.train_size)
		init_lr = FLAGS.init_lr

		distillation_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))

		if FLAGS.use_tpu:
			warmup_ratio = config.get('warmup', 0.1)

			num_train_steps = int(
				train_size / FLAGS.batch_size * FLAGS.epoch)
			
			num_warmup_steps = int(num_train_steps * warmup_ratio)
			
			print('==num warmup steps==', num_warmup_steps)

			print(" model type {}".format(FLAGS.model_type))

			print(num_train_steps, num_warmup_steps, "=============", kargs.get('num_gpus', 1), '==number of gpus==')
			tf.logging.info("***** Running evaluation *****")
			tf.logging.info("***** train steps : %d", num_train_steps)
			max_eval_steps = int(int(FLAGS.eval_size) / FLAGS.batch_size)

			clip_norm_scale = 1.0
			lr_scale = 1.0
			lr = init_lr

			checkpoint_dir = checkpoint_dir

			opt_config = Bunch({"init_lr":lr, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"train_op":kargs.get("train_op", "adam"),
							"decay":kargs.get("decay", "no"),
							"warmup":kargs.get("warmup", "no"),
							"clip_norm":config.get("clip_norm", 1.0),
							"grad_clip":config.get("grad_clip", "global_norm"),
							"use_tpu":1})

		else:
			warmup_ratio = config.get('warmup', 0.1)
			worker_count = kargs.get('worker_count', 1)
			task_index = kargs.get('task_index', 0)
			is_chief = kargs.get('is_chief', 0)

			if FLAGS.if_shard == "0":
				train_size = FLAGS.train_size
				epoch = int(FLAGS.epoch / worker_count)
			elif FLAGS.if_shard == "1":
				print("==number of gpus==", kargs.get('num_gpus', 1))
				train_size = int(FLAGS.train_size/worker_count/kargs.get('num_gpus', 1))
				# train_size = int(FLAGS.train_size)
				epoch = FLAGS.epoch
			else:
				train_size = int(FLAGS.train_size/worker_count)
				epoch = FLAGS.epoch

			num_train_steps = int(
				train_size / FLAGS.batch_size * epoch)
			if config.get('ln_type', 'postln') == 'postln':
				num_warmup_steps = int(num_train_steps * warmup_ratio)
			elif config.get('ln_type', 'preln') == 'postln':
				num_warmup_steps = 0
			else:
				num_warmup_steps = int(num_train_steps * warmup_ratio)
			print('==num warmup steps==', num_warmup_steps)

			num_storage_steps = min([int(train_size / FLAGS.batch_size), 10000 ])
			if num_storage_steps <= 100:
				num_storage_steps = 500

			num_eval_steps = int(FLAGS.eval_size / FLAGS.batch_size)

			print("num_train_steps {}, num_eval_steps {}, num_storage_steps {}".format(num_train_steps, num_eval_steps, num_storage_steps))

			print(" model type {}".format(FLAGS.model_type))

			print(num_train_steps, num_warmup_steps, "=============", kargs.get('num_gpus', 1), '==number of gpus==')

			if worker_count*kargs.get("num_gpus", 1) >= 2:
				clip_norm_scale = 1.0
				lr_scale = 0.8
			else:
				clip_norm_scale = 1.0
				lr_scale = 1.0
			lr = init_lr*worker_count*kargs.get("num_gpus", 1)*lr_scale
			if lr >= 1e-3:
				lr = 1e-3
			print('==init lr==', lr)
			if FLAGS.opt_type == "hvd" and hvd:
				checkpoint_dir = checkpoint_dir if task_index == 0 else None
			elif FLAGS.opt_type == "all_reduce":
				checkpoint_dir = checkpoint_dir
			elif FLAGS.opt_type == "collective_reduce":
				checkpoint_dir = checkpoint_dir if task_index == 0 else None
			elif FLAGS.opt_type == "ps" or FLAGS.opt_type == "ps_sync":
				checkpoint_dir = checkpoint_dir if task_index == 0 else None

			opt_config = Bunch({"init_lr":lr, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"worker_count":worker_count,
							"gpu_count":worker_count*kargs.get("num_gpus", 1),
							"opt_type":FLAGS.opt_type,
							"is_chief":is_chief,
							"train_op":kargs.get("train_op", "adam"),
							"decay":kargs.get("decay", "no"),
							"warmup":kargs.get("warmup", "no"),
							"clip_norm":config.get("clip_norm", 1.0),
							"grad_clip":config.get("grad_clip", "global_norm"),
							"epoch":FLAGS.epoch,
							"strategy":FLAGS.distribution_strategy,
							"use_tpu":0})

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
			load_pretrained_dict[task_type] = distillation_config[task_type]["load_pretrained"]
			exclude_scope_dict[task_type] = distillation_config[task_type]["exclude_scope"]
			not_storage_params_dict[task_type] = distillation_config[task_type]["not_storage_params"]
			target_dict[task_type] = distillation_config[task_type]["target"]

		tf.logging.info("***** use tpu ***** %s", str(FLAGS.use_tpu))

		if kargs.get('electra_mode', 'solo_training') == 'solo_training':
			tf.logging.info("***** electra mode ***** generator and discriminator seperate training")
			model_fn_builder = classifier_model_fn_builder
		elif kargs.get('electra_mode', 'solo_training') == 'gumbel_training':
			tf.logging.info("***** electra mode ***** gumbel generator and discriminator joint training")
			model_fn_builder = classifier_model_fn_builder_gumbel
		else:
			model_fn_builder = classifier_model_fn_builder

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

		if FLAGS.use_tpu:
			from data_generator import tf_data_utils
			estimator = tf.contrib.tpu.TPUEstimator(
					  use_tpu=True,
					  model_fn=model_fn,
					  config=kargs.get('run_config', {}),
					  train_batch_size=FLAGS.batch_size,
					  eval_batch_size=FLAGS.batch_size)
			tf.logging.info("****** do train ******* %s", str(FLAGS.do_train))
			if FLAGS.do_train:
				tf.logging.info("***** Running training *****")
				tf.logging.info("  Batch size = %d", FLAGS.batch_size)
				input_features = tf_data_utils.electra_input_fn_builder(train_file, 
											FLAGS.max_length,
											FLAGS.max_predictions_per_seq,
											True,
											num_cpu_threads=4)
				estimator.train(input_fn=input_features, max_steps=num_train_steps)
			else:
				tf.logging.info("***** Running evaluation *****")
				tf.logging.info("  Batch size = %d", FLAGS.batch_size)
				eval_input_fn = tf_data_utils.electra_input_fn_builder(
								input_files=dev_file,
								max_seq_length=FLAGS.max_length,
								max_predictions_per_seq=FLAGS.max_predictions_per_seq,
								is_training=False)
				tf.logging.info("***** Begining Running evaluation *****")
				result = estimator.evaluate(input_fn=eval_input_fn, steps=max_eval_steps)
				output_eval_file = os.path.join(checkpoint_dir, "eval_results.txt")
				with tf.gfile.GFile(output_eval_file, "w") as writer:
					tf.logging.info("***** Eval results *****")
					for key in sorted(result.keys()):
						tf.logging.info("  %s = %s", key, str(result[key]))
						writer.write("%s = %s\n" % (key, str(result[key])))
		else:
			from data_generator import distributed_tf_data_utils as tf_data_utils
			name_to_features = {
				"input_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_ori_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"masked_lm_positions":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
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

			if kargs.get("run_config", None):
				if kargs.get("parse_type", "parse_single") == "parse_single":
					train_features = lambda: tf_data_utils.all_reduce_train_input_fn(train_file,
												_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
												worker_count=worker_count,
												task_index=task_index)
					eval_features = lambda: tf_data_utils.all_reduce_eval_input_fn(dev_file,
												_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
												worker_count=worker_count,
												task_index=task_index)
				elif kargs.get("parse_type", "parse_single") == "parse_batch":
					print("==apply parse example==")
					train_features = lambda: tf_data_utils.all_reduce_train_batch_input_fn(train_file,
												_decode_batch_record, name_to_features, params, if_shard=FLAGS.if_shard,
												worker_count=worker_count,
												task_index=task_index)
					eval_features = lambda: tf_data_utils.all_reduce_eval_batch_input_fn(dev_file,
												_decode_batch_record, name_to_features, params, if_shard=FLAGS.if_shard,
												worker_count=worker_count,
												task_index=task_index)

			else:
				train_features = lambda: tf_data_utils.train_input_fn(train_file,
											_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
											worker_count=worker_count,
											task_index=task_index)

				eval_features = lambda: tf_data_utils.eval_input_fn(dev_file,
											_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
											worker_count=worker_count,
											task_index=task_index)
			
			train_hooks = []
			eval_hooks = []

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
							model_dir=checkpoint_dir,
							config=run_config)

			train_being_time = time.time()
			tf.logging.info("==training distribution_strategy=={}".format(kargs.get("distribution_strategy", "MirroredStrategy")))
			if kargs.get("distribution_strategy", "MirroredStrategy") == "MirroredStrategy":
				print("==apply single machine multi-card training==")

				train_spec = tf.estimator.TrainSpec(input_fn=train_features, 
												max_steps=num_train_steps)

				eval_spec = tf.estimator.EvalSpec(input_fn=eval_features, 
												steps=num_eval_steps)
				
				model_estimator.train(input_fn=train_features,
								max_steps=num_train_steps,
								hooks=train_hooks)
				# tf.estimator.train(model_estimator, train_spec)

				train_end_time = time.time()
				print("==training time==", train_end_time - train_being_time)
				tf.logging.info("==training time=={}".format(train_end_time - train_being_time))
				eval_results = model_estimator.evaluate(input_fn=eval_features, steps=num_eval_steps)
				print(eval_results)
				
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

				# tf.estimator.train(model_estimator, train_spec) # tf 1.12 doesn't need evaluate

				tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
