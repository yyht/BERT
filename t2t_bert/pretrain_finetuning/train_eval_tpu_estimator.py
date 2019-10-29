# -*- coding: utf-8 -*-
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import distributed_tf_data_utils as tf_data_utils

try:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface
except:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface

try:
	from .classifier_fn_tpu_estimator import classifier_model_fn_builder
except:
	from classifier_fn_tpu_estimator import classifier_model_fn_builder

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

		warmup_ratio = config.get('warmup', 0.1)

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		
		num_warmup_steps = int(num_train_steps * warmup_ratio)
		
		print('==num warmup steps==', num_warmup_steps)

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============", kargs.get('num_gpus', 1), '==number of gpus==')

		max_eval_steps = int(int(FLAGS.eval_size) / FLAGS.batch_size)

		clip_norm_scale = 1.0
		lr_scale = 1.0
		lr = init_lr
		
		opt_config = Bunch({"init_lr":lr, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps,
							"train_op":kargs.get("train_op", "adam"),
							"decay":kargs.get("decay", "no"),
							"warmup":kargs.get("warmup", "no"),
							"clip_norm":config.get("clip_norm", 1.0),
							"grad_clip":config.get("grad_clip", "global_norm"),
							"use_tpu":1})

		model_io_config = Bunch({"fix_lm":False})
		model_io_fn = model_io.ModelIO(model_io_config)
		
		num_classes = FLAGS.num_classes
		checkpoint_dir = checkpoint_dir

		model_fn = classifier_model_fn_builder(config, 
									num_classes, 
									init_checkpoint, 
									model_reuse=None, 
									load_pretrained=FLAGS.load_pretrained,
									model_io_config=model_io_config,
									opt_config=opt_config,
									model_io_fn=model_io_fn,
									exclude_scope="",
									not_storage_params=[],
									target=kargs.get("input_target", ""),
									**kargs)

		input_features = tf_data_utils.input_fn_builder(train_file, 
										FLAGS.max_length,
										FLAGS.max_predictions_per_seq,
										FLAGS.do_train,
										num_cpu_threads=4)

		estimator = tf.contrib.tpu.TPUEstimator(
				  use_tpu=FLAGS.use_tpu,
				  model_fn=model_fn,
				  config=kargs.get('run_config', {}),
				  train_batch_size=FLAGS.batch_size,
				  eval_batch_size=FLAGS.batch_size)

		if FLAGS.do_train:
			estimator.train(input_fn=input_features, max_steps=num_train_steps)
		else:
			result = estimator.evaluate(input_fn=input_features, steps=max_eval_steps)
			output_eval_file = os.path.join(checkpoint_dir, "eval_results.txt")
			with tf.gfile.GFile(output_eval_file, "w") as writer:
				tf.logging.info("***** Eval results *****")
				for key in sorted(result.keys()):
					tf.logging.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))
