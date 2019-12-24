import re
import tensorflow as tf
from optimizer import adam_weight_decay_exclude_utils
from optimizer import optimizer_utils

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
					**kargs):

	gpu_count = kargs.get('gpu_count', 1)

	global_step = tf.train.get_or_create_global_step()
	learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

	# Implements linear decay of the learning rate.
	learning_rate = tf.train.polynomial_decay(
	  learning_rate,
	  global_step,
	  num_train_steps,
	  end_learning_rate=0.0,
	  power=1.0,
	  cycle=False)

	if num_warmup_steps:
		global_steps_int = tf.cast(global_step, tf.int32)
		warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

		global_steps_float = tf.cast(global_steps_int, tf.float32)
		warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

		warmup_percent_done = global_steps_float / warmup_steps_float
		warmup_learning_rate = init_lr * warmup_percent_done

		is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
		learning_rate = (
			(1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

	if kargs.get('optimizer_type', 'inner_adamw') == 'inner_adamw':
		optimizer = adam_weight_decay_exclude_utils.AdamWOptimizer(
							weight_decay=0.01,
							learning_rate=learning_rate,
							beta1=0.9,
							beta2=0.999,
							epsilon=1e-6,
							exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
							)
	elif kargs.get('optimizer_type', 'tpu_adamw') == 'inner_adamw':
		optimizer = optimizer_utils.AdamWeightDecayOptimizer(
		  learning_rate=learning_rate,
		  weight_decay_rate=0.01,
		  beta_1=0.9,
		  beta_2=0.999,
		  epsilon=1e-6,
		  exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
	else:
		optimizer = optimizer_utils.AdamWeightDecayOptimizer(
		  learning_rate=learning_rate,
		  weight_decay_rate=0.01,
		  beta_1=0.9,
		  beta_2=0.999,
		  epsilon=1e-6,
		  exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
	tvars = tf.trainable_variables()

	if use_tpu:
		optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

	if kargs.get('optimizer_type', 'inner_adamw') == 'inner_adamw':
		grads_and_vars = optimizer.compute_gradients(loss, tvars)
		grads = [grad/gpu_count for grad, _ in grads_and_vars]

		[scale_grads, _] = tf.clip_by_global_norm(grads, 
										clip_norm=1.0)

		grads_and_vars = zip(scale_grads, tvars)
		train_op = optimizer.apply_gradients(
					grads_and_vars, global_step=global_step)
	elif kargs.get('optimizer_type', 'tpu_adamw') == 'inner_adamw':
		grads = tf.gradients(loss, tvars)
		(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
		train_op = optimizer.apply_gradients(
						zip(grads, tvars), global_step=global_step)

		new_global_step = global_step + 1
		train_op = tf.group(train_op, [global_step.assign(new_global_step)])
	else:
		grads = tf.gradients(loss, tvars)
		(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
		train_op = optimizer.apply_gradients(
						zip(grads, tvars), global_step=global_step)

		new_global_step = global_step + 1
		train_op = tf.group(train_op, [global_step.assign(new_global_step)])
	return train_op






