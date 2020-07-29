# -*- coding: utf-8 -*-
import tensorflow as tf

from optimizer import distributed_optimizer as optimizer
from data_generator import tf_data_utils, tf_pretrain_data_utils

try:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface
except:
	from distributed_single_sentence_classification.model_interface import model_config_parser
	from distributed_single_sentence_classification.model_data_interface import data_interface

try:
	from .classifier_fn_tpu_estimator import classifier_model_fn_builder
	from .classifier_fn_tpu_bert_seq_estimator import classifier_model_fn_builder as classifier_seq_model_fn_builder
	from .classifier_fn_tpu_gatedcnn_estimator import classifier_model_fn_builder as gatedcnn_model_fn_builder
except:
	from classifier_fn_tpu_estimator import classifier_model_fn_builder
	from classifier_fn_tpu_bert_seq_estimator import classifier_model_fn_builder as classifier_seq_model_fn_builder
	from classifier_fn_tpu_gatedcnn_estimator import classifier_model_fn_builder as gatedcnn_model_fn_builder

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

		if FLAGS.model_type == 'bert':
			model_fn_builder = classifier_model_fn_builder
			tf.logging.info("****** bert mlm ******")
		elif FLAGS.model_type == 'bert_seq':
			model_fn_builder = classifier_seq_model_fn_builder
			tf.logging.info("****** bert seq as gpt ******")
		elif FLAGS.model_type == 'gated_cnn_seq':
			model_fn_builder = gatedcnn_model_fn_builder
			tf.logging.info("****** gated cnn seq ******")
		else:
			model_fn_builder = classifier_model_fn_builder
			tf.logging.info("****** bert mlm ******")

		model_fn = model_fn_builder(config, 
									num_classes, 
									init_checkpoint, 
									model_reuse=None, 
									load_pretrained=FLAGS.load_pretrained,
									model_io_config=model_io_config,
									opt_config=opt_config,
									model_io_fn=model_io_fn,
									# exclude_scope=kargs.get('exclude_scope', ""),
									not_storage_params=[],
									target=kargs.get("input_target", ""),
									num_train_steps=num_train_steps,
									use_tpu=True,
									**kargs)

		estimator = tf.contrib.tpu.TPUEstimator(
				  use_tpu=True,
				  model_fn=model_fn,
				  config=kargs.get('run_config', {}),
				  train_batch_size=FLAGS.batch_size,
				  eval_batch_size=FLAGS.batch_size)
		tf.logging.info("****** do train ******* %s", str(FLAGS.do_train))

		if FLAGS.random_generator == "1":
			input_fn_builder = tf_data_utils.electra_input_fn_builder
			tf.logging.info("***** Running random sample input fn builder *****")
		elif FLAGS.random_generator == "2":
			input_fn_builder = tf_data_utils.bert_seq_input_fn_builder
			tf.logging.info("***** Running bert seq input fn builder *****")
		elif FLAGS.random_generator == "3":
			input_fn_builder = tf_data_utils.bert_mnli_input_fn_builder
			tf.logging.info("***** Running bert seq input fn builder *****")
		elif FLAGS.random_generator == "4":
			input_fn_builder = tf_data_utils.gatedcnn_pretrain_input_fn_builder_v1
			tf.logging.info("***** Running gatedcnn input fn builder *****")
		elif FLAGS.random_generator == "5":
			input_fn_builder = tf_pretrain_data_utils.input_fn_builder
			data_config = Bunch({})
			data_config.min_tok = 1
			data_config.max_tok = 10
			data_config.sep_id = 102
			data_config.pad_id = 0
			data_config.cls_id = 101
			data_config.mask_id = 103
			data_config.leak_ratio = 0.0
			data_config.rand_ratio = 1.0
			data_config.vocab_size = config.vocab_size
			data_config.mask_prob = 0.25
			data_config.sample_strategy = 'token_span'
			data_config.truncate_seq = False
			data_config.stride = 1
			data_config.use_bfloat16 = False
			tf.logging.info("***** Running efficiency input fn builder *****")
		else:
			input_fn_builder = tf_data_utils.input_fn_builder
			tf.logging.info("***** Running fixed sample input fn builder *****")

		if FLAGS.do_train:
			tf.logging.info("***** Running training *****")
			tf.logging.info("  Batch size = %d", FLAGS.batch_size)
			input_features = input_fn_builder(train_file, 
										FLAGS.max_length,
										FLAGS.max_predictions_per_seq,
										True,
										num_cpu_threads=4,
										FLAGS=data_config,
										truncate_seq=data_config.truncate_seq, 
										use_bfloat16=data_config.use_bfloat16,
										stride=data_config.stride)
			estimator.train(input_fn=input_features, max_steps=num_train_steps)
		else:
			tf.logging.info("***** Running evaluation *****")
			tf.logging.info("  Batch size = %d", FLAGS.batch_size)
			eval_input_fn = input_fn_builder(
							input_files=dev_file,
							max_seq_length=FLAGS.max_length,
							max_predictions_per_seq=FLAGS.max_predictions_per_seq,
							is_training=False,
							num_cpu_threads=4,
							FLAGS=data_config,
							truncate_seq=data_config.truncate_seq, 
							use_bfloat16=data_config.use_bfloat16,
							stride=data_config.stride)
			tf.logging.info("***** Begining Running evaluation *****")
			result = estimator.evaluate(input_fn=eval_input_fn, steps=max_eval_steps)
			output_eval_file = os.path.join(checkpoint_dir, "eval_results.txt")
			with tf.gfile.GFile(output_eval_file, "w") as writer:
				tf.logging.info("***** Eval results *****")
				for key in sorted(result.keys()):
					tf.logging.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))
