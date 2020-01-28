import tensorflow as tf
import numpy as np

def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
	training_progress = tf.to_float(global_step) / tf.to_float(num_train_steps)
	if schedule == "linear_schedule":
		threshold = training_progress
	elif schedule == "exp_schedule":
		scale = 5
		threshold = tf.exp((training_progress - 1) * scale)
		# [exp(-5), exp(0)] = [1e-2, 1]
	elif schedule == "log_schedule":
		scale = 5
	# [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
		threshold = 1 - tf.exp((-training_progress) * scale)
	return threshold * (end - start) + start