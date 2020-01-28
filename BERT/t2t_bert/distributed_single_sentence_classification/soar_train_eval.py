# -*- coding: utf-8 -*-
import sys,os

import tensorflow as tf
import os
try:
	from .train_eval_estimator_fn import train_eval_fn as estimator_fn
	from .train_eval_sess_fn import train_eval_fn as sess_fn
	from .eval_estimator_fn import eval_fn as estimator_eval_fn
	from .eval_sess_fn import eval_fn as sess_eval_fn
except:
	from train_eval_estimator_fn import train_eval_fn as estimator_fn
	from train_eval_sess_fn import train_eval_fn as sess_fn
	from eval_estimator_fn import eval_fn as estimator_eval_fn
	from eval_sess_fn import eval_fn as sess_eval_fn

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
		print("==begin to train==")
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
		print("==begin to eval==")
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

def monitored_sess(FLAGS,
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
	print("==begin to eval==")
	if kargs.get("running_type", "train") == "eval":
		result_dict = sess_eval_fn(FLAGS,
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
		return result_dict
	elif kargs.get("running_type", "train") == "train":
		sess_fn(FLAGS,
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