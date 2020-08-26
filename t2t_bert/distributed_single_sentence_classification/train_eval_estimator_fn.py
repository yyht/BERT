# -*- coding: utf-8 -*-
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils
from data_generator import tf_pretrain_data_utils
from data_generator import tf_data_utils_confusion_set


# try:
# 	from .bert_model_fn import model_fn_builder
# 	from .bert_model_fn import rule_model_fn_builder
# 	from .classifynet_model_fn import 
# except:
# 	from bert_model_fn import model_fn_builder
# 	from bert_model_fn import rule_model_fn_builder

# try:
# 	from .model_fn import model_fn_builder
# 	from .model_interface import model_config_parser
# 	from .model_data_interface import data_interface
# 	from .model_distillation_fn import model_fn_builder as model_distillation_fn
# except:
# 	from model_fn import model_fn_builder
# 	from model_interface import model_config_parser
# 	from model_data_interface import data_interface
# 	from model_distillation_fn import model_fn_builder as model_distillation_fn

try:
	# from .model_fn import model_fn_builder
	from .model_interface import model_config_parser
	from .model_data_interface import data_interface
	from .model_fn_interface import model_fn_interface
	# from .model_distillation_fn import model_fn_builder as model_distillation_fn
except:
	# from model_fn import model_fn_builder
	from model_interface import model_config_parser
	from model_data_interface import data_interface
	# from model_distillation_fn import model_fn_builder as model_distillation_fn
	from model_fn_interface import model_fn_interface

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
				
		# config = json.load(open(FLAGS.config_file, "r"))

		# config = Bunch(config)
		# config.use_one_hot_embeddings = True
		# config.scope = "bert"
		# config.dropout_prob = 0.1
		# config.label_type = "single_label"

		# config.model = FLAGS.model_type

		config = model_config_parser(FLAGS)
		
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

		init_lr = config.init_lr

		# label_dict = json.load(tf.gfile.Open(FLAGS.label_id))

		warmup_ratio = config.get('warmup', 0.1)

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

		if is_debug == "0":
			num_storage_steps = 2
			num_eval_steps = 10
			num_train_steps = 10
		print("num_train_steps {}, num_eval_steps {}, num_storage_steps {}".format(num_train_steps, num_eval_steps, num_storage_steps))

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============", kargs.get('num_gpus', 1), '==number of gpus==')

		# if worker_count*kargs.get("num_gpus", 1) >= 2:
		# 	clip_norm_scale = 1.0
		# 	lr_scale = 0.8
		# else:
		# 	clip_norm_scale = 1.0
		# 	lr_scale = 1.0
		# lr = init_lr*worker_count*kargs.get("num_gpus", 1)*lr_scale
		# if lr >= 1e-3:
		# 	lr = 1e-3
		lr = init_lr
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
							"beta_2":0.99,
							"strategy":FLAGS.distribution_strategy,
							"use_tpu":False})

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

		# if kargs.get("rule_model", "normal") == "rule":
		# 	model_fn_interface = rule_model_fn_builder
		# 	print("==apply rule model==")
		# else:
		# 	model_fn_interface = model_fn_builder
		# 	print("==apply normal model==")

		model_fn_builder = model_fn_interface(FLAGS)
		print("==use-tpu==", FLAGS.use_tpu)

		model_fn = 	model_fn_builder(config, num_classes, init_checkpoint, 
									model_reuse=None, 
									load_pretrained=FLAGS.load_pretrained,
									model_io_config=model_io_config,
									opt_config=opt_config,
									model_io_fn=model_io_fn,
									exclude_scope=FLAGS.exclude_scope,
									not_storage_params=[],
									target=kargs.get("input_target", ""),
									output_type="estimator",
									checkpoint_dir=checkpoint_dir,
									num_storage_steps=num_storage_steps,
									task_index=task_index,
									anneal_config=anneal_config,
									use_tpu=FLAGS.use_tpu,
									**kargs)
		print(model_fn, "===model-fn====")

		name_to_features = data_interface(FLAGS)

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
			elif kargs.get("parse_type", "parse_single") == "parse_batch_multi_task":
				data_prior = [float(item) for item in FLAGS.data_prior.split(',')]
				train_features = lambda: tf_data_utils.all_reduce_multitask_train_batch_input_fn_sample(
										train_file,
										_decode_record, 
										name_to_features, 
										params, 
										data_prior=data_prior,
										if_shard=FLAGS.if_shard,
										worker_count=worker_count,
										task_index=task_index)
				eval_features = lambda: tf_data_utils.all_reduce_eval_batch_input_fn(dev_file,
											_decode_batch_record, name_to_features, params, if_shard=FLAGS.if_shard,
											worker_count=worker_count,
											task_index=task_index)
			elif kargs.get("parse_type", "parse_dynamic") == 'parse_dynamic':
				
				try:
					[confusion_matrix,
					confusion_mask_matrix] = tf_data_utils_confusion_set.load_confusion_set(FLAGS.confusion_set_path,
																FLAGS.confusion_set_mask_path)
				# confusion_matrix = tf.convert_to_tensor(confusion_matrix, dtype=tf.int32)
				# confusion_mask_matrix = tf.convert_to_tensor(confusion_mask_matrix, dtype=tf.int32)
					tf.logging.info("***** Running confusion set sampling *****")
				except:
					confusion_matrix = None
					confusion_mask_matrix = None
					tf.logging.info("***** Running random sampling *****")

				data_config = Bunch({})
				data_config.confusion_matrix = confusion_matrix
				data_config.confusion_mask_matrix = confusion_mask_matrix
				data_config.min_tok = 1
				data_config.max_tok = 10
				data_config.sep_id = 102
				data_config.pad_id = 0
				data_config.cls_id = 101
				data_config.mask_id = 103
				data_config.leak_ratio = 0.1
				data_config.rand_ratio = 0.5
				data_config.vocab_size = config.vocab_size
				data_config.mask_prob = 0.15
				data_config.sample_strategy = 'token_span'
				data_config.truncate_seq = False
				data_config.stride = 1
				data_config.use_bfloat16 = False
				tf.logging.info("***** Running efficiency input fn builder *****")
				train_features = tf_pretrain_data_utils.input_fn_builder(train_file, 
										FLAGS.max_length,
										FLAGS.max_predictions_per_seq,
										True,
										num_cpu_threads=4,
										FLAGS=data_config,
										truncate_seq=data_config.truncate_seq, 
										use_bfloat16=data_config.use_bfloat16,
										stride=data_config.stride)
				eval_features = lambda: tf_data_utils.all_reduce_eval_input_fn(dev_file,
											_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
											worker_count=worker_count,
											task_index=task_index)
				print("===using parse_dynamic generator online===")
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
						params=params,
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

			def estimator_eval_fn(ckpt_path):
				return model_estimator.evaluate(
				              input_fn=eval_features,
				              steps=num_eval_steps,
				              checkpoint_path=ckpt_path)

			try:
				from .evaluate_all_ckpt import evalue_all_ckpt
			except:
				from evaluate_all_ckpt import evalue_all_ckpt

			evalue_all_ckpt(checkpoint_dir, "classification", estimator_eval_fn)
		
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
			# train_end_time = time.time()
			# print("==training time==", train_end_time - train_being_time)

		
		