# -*- coding: utf-8 -*-
import sys,os

import tensorflow as tf
import os
try:
	from .train_eval_gpu_electra_estimator import train_eval_fn as estimator_fn
	# from .train_eval_sess_fn import train_eval_fn as sess_fn
	# from .eval_sess_fn import eval_fn as sess_eval_fn
except:
	from train_eval_gpu_electra_estimator import train_eval_fn as estimator_fn
	# from train_eval_sess_fn import train_eval_fn as sess_fn
	# from eval_sess_fn import eval_fn as sess_eval_fn

def monitored_estimator(
				FLAGS,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir,
				**kargs):

	if kargs.get("running_type", "train") == "train":
		print("==begin to train==")
		estimator_fn(FLAGS=FLAGS,
					init_checkpoint=init_checkpoint,
					train_file=train_file,
					dev_file=dev_file,
					checkpoint_dir=checkpoint_dir,
					**kargs)