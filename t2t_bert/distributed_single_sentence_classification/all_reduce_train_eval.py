# -*- coding: utf-8 -*-
import sys,os

import tensorflow as tf
import os
try:
	from .train_eval_estimator_fn import train_eval_fn as estimator_fn
	from .eval_estimator_fn import train_eval_fn as estimator_eval_fn
except:
	from train_eval_estimator_fn import train_eval_fn as estimator_fn
	from eval_estimator_fn import train_eval_fn as estimator_eval_fn

def monitored_estimator(FLAGS,
				worker_count, 
				task_index, 
				cluster, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir,
				**kargs):

	if kargs.get("running_type", "train") == "train":
		estimator_fn(FLAGS,
					worker_count, 
					task_index, 
					is_chief, 
					target,
					init_checkpoint,
					train_file,
					dev_file,
					checkpoint_dir,
					FLAGS.is_debug,
					**kargs)
	elif kargs.get("running_type", "eval") == "eval":
		estimator_eval_fn(FLAGS,
					worker_count, 
					task_index, 
					is_chief, 
					target,
					init_checkpoint,
					train_file,
					dev_file,
					checkpoint_dir,
					FLAGS.is_debug,
					**kargs)

