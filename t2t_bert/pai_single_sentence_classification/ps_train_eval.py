# -*- coding: utf-8 -*-
import sys,os

import tensorflow as tf
import os
from .train_eval_sess_fn import train_eval_fn as sess_fn
from .train_eval_estimator_fn import train_eval_fn as estimator_fn

def monitored_sess(FLAGS,
				worker_count, 
				task_index, 
				cluster, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir):

	if worker_count >= 1 and FLAGS.opt_type == "ps_sync" or FLAGS.opt_type == "ps":
		print("==starting parameter server distributed traiing==")
		available_worker_device = "/job:worker/task:%d" % (task_index)
		with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
			sess_fn(FLAGS,
							worker_count, 
							task_index, 
							is_chief, 
							target,
							init_checkpoint,
							train_file,
							dev_file,
							checkpoint_dir,
							FLAGS.is_debug)
	else:
		sess_fn(FLAGS,
					worker_count, 
					task_index, 
					is_chief, 
					target,
					init_checkpoint,
					train_file,
					dev_file,
					checkpoint_dir,
					FLAGS.is_debug)

def monitored_estimator(FLAGS,
				worker_count, 
				task_index, 
				cluster, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir):
	
	estimator_fn(FLAGS,
					worker_count, 
					task_index, 
					is_chief, 
					target,
					init_checkpoint,
					train_file,
					dev_file,
					checkpoint_dir,
					FLAGS.is_debug)
	
