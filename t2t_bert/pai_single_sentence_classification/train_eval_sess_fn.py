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

		model_train_fn = model_fn_builder(config, num_classes, init_checkpoint, 
												model_reuse=None, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config,
												exclude_scope="",
												not_storage_params=[],
												target="")
		
		model_eval_fn = model_fn_builder(config, num_classes, init_checkpoint, 
												model_reuse=True, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config,
												exclude_scope="",
												not_storage_params=[],
												target="")
		if FLAGS.opt_type == "ps":
			sync_replicas_hook = optimizer_fn.opt.make_session_run_hook(is_chief, num_tokens=0)
		else:
			sync_replicas_hook = []
		
		def eval_metric_fn(features, eval_op_dict):
			logits = eval_op_dict["logits"]
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			accuracy = correct = tf.equal(
				tf.cast(pred_label, tf.int32),
				tf.cast(features["label_ids"], tf.int32)
			)
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

			return {"accuracy":accuracy, "loss":eval_op_dict["loss"], 
					"pred_label":pred_label, "label_ids":features["label_ids"]}

		def train_metric_fn(features, train_op_dict):
			logits = train_op_dict["logits"]
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			accuracy = correct = tf.equal(
				tf.cast(pred_label, tf.int32),
				tf.cast(features["label_ids"], tf.int32)
			)
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
			return {"accuracy":accuracy, "loss":train_op_dict["loss"], 
					"train_op":train_op_dict["train_op"]}
		
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

		train_features = tf_data_utils.train_input_fn(train_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)

		eval_features = tf_data_utils.eval_input_fn(dev_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard,
									worker_count=worker_count,
									task_index=task_index)
		
		train_op_dict = model_train_fn(train_features, [], tf.estimator.ModeKeys.TRAIN)
		eval_op_dict = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
		eval_dict = eval_metric_fn(eval_features, eval_op_dict["eval"])
		train_dict = train_metric_fn(train_features, train_op_dict["train"])

		print("===========begin to train============")
		sess_config = tf.ConfigProto(allow_soft_placement=False,
									log_device_placement=False)

		checkpoint_dir = checkpoint_dir if task_index == 0 else None

		print("start training") 

		# hooks = [tf.train.StopAtStepHook(last_step=num_train_steps)]
		hooks = []
		if FLAGS.opt_type == "ps":
			sync_replicas_hook = optimizer_fn.opt.make_session_run_hook(is_chief, num_tokens=0)
			hooks.append(sync_replicas_hook)
			sess = tf.train.MonitoredTrainingSession(master=target,
												 is_chief=is_chief,
												 config=sess_config,
												 hooks=[],
												 checkpoint_dir=checkpoint_dir,
												 save_checkpoint_steps=num_storage_steps)
		else:
			sess = tf.train.MonitoredTrainingSession(config=sess_config,
                                           hooks=[],
                                           checkpoint_dir=checkpoint_dir,
										   save_checkpoint_steps=num_storage_steps)
		
		def eval_fn(eval_dict, sess):
			i = 0
			total_accuracy = 0
			eval_total_dict = {}
			while True:
				try:
					eval_result = sess.run(eval_dict)
					for key in eval_result:
						if key not in eval_total_dict:
							if key in ["pred_label", "label_ids"]:
								eval_total_dict[key] = []
								eval_total_dict[key].extend(eval_result[key])
							if key in ["accuracy", "loss"]:
								eval_total_dict[key] = 0.0
								eval_total_dict[key] += eval_result[key]
						else:
							if key in ["pred_label", "label_ids"]:
								eval_total_dict[key].extend(eval_result[key])
							if key in ["accuracy", "loss"]:
								eval_total_dict[key] += eval_result[key]

					i += 1
					if np.mod(i, num_eval_steps) == 0:
						break
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break

			label_id = eval_total_dict["label_ids"]
			pred_label = eval_total_dict["pred_label"]

			result = classification_report(label_id, pred_label, 
				target_names=list(label_dict["label2id"].keys()))

			print(result, task_index)
			eval_total_dict["classification_report"] = result
			return eval_total_dict

		def train_fn(train_op_dict, sess):
			i = 0
			cnt = 0
			loss_dict = {}
			monitoring_train = []
			monitoring_eval = []
			while True:
				try:
					[train_result] = sess.run([train_op_dict])
					step = sess.run(tf.train.get_global_step())
					for key in train_result:
						if key == "train_op":
							continue
						else:
							if np.isnan(train_result[key]):
								print(train_loss, "get nan loss")
								break
							else:
								if key in loss_dict:
									loss_dict[key] += train_result[key]
								else:
									loss_dict[key] = train_result[key]
					
					i += 1
					cnt += 1
					
					if np.mod(i, num_storage_steps) == 0:
						string = ""
						for key in loss_dict:
							tmp = key + " " + str(loss_dict[key]/cnt) + "\t"
							string += tmp
						print(string, step)
						monitoring_train.append(loss_dict)

						eval_finial_dict = eval_fn(eval_dict, sess)
						monitoring_eval.append(eval_finial_dict)

						for key in loss_dict:
							loss_dict[key] = 0.0
						cnt = 0

					if is_debug == "0":
						if i == num_train_steps:
							break

				except tf.errors.OutOfRangeError:
					print("==Succeeded in training model==")
						
		# print("===========begin to train============")
		# sess_config = tf.ConfigProto(allow_soft_placement=False,
		# 							log_device_placement=False)

		# checkpoint_dir = checkpoint_dir if task_index == 0 else None

		# print("start training") 

		# hooks = [tf.train.StopAtStepHook(last_step=num_train_steps)]
		# if sync_replicas_hook:
		# 	hooks.append(sync_replicas_hook)

		# sess = tf.train.MonitoredTrainingSession(master=target,
		# 									 is_chief=is_chief,
		# 									 config=sess_config,
		# 									 hooks=[],
		# 									 checkpoint_dir=checkpoint_dir,
		# 									 save_checkpoint_steps=num_storage_steps)

		# with tf.train.MonitoredTrainingSession(master=target,
		# 									 is_chief=is_chief,
		# 									 config=sess_config,
		# 									 hooks=[],
		# 									 checkpoint_dir=checkpoint_dir,
		# 									 save_checkpoint_steps=num_storage_steps) as sess:
		step = sess.run(optimizer_fn.global_step)
		print(step)
		train_fn(train_dict, sess)

		if task_index == 0:
			print("===========begin to eval============")
			eval_finial_dict = eval_fn(eval_dict, sess)