# prev_op = tf.no_op()

		# loop_step_dict = kargs.get('loop_step_dict', None)
		# if not loop_step_dict:
		# 	loop_step_dict = {}
		# 	for key in loss_dict:
		# 		loop_step_dict[key] = 1

		# optimizer_dict = {}

		# # global_step_dict = kargs.get('global_step_dict', None)
		# fce_acc = kargs.get("fce_acc", None)
		# num_train_steps = kargs.get('num_train_steps', 100000)

		# # for key in init_lr_dict:
		# # 	init_lr = init_lr_dict[key]
		# # 	optimizer_type = optimizer_type_dict[key]
		# # 	if optimizer_type != 'radam':
		# # 		learning_rate = self.private_lr_decay_fn(init_lr, num_train_steps,
		# # 												global_step_dict[key], **kargs)
		# # 		learning_rate = self.private_warm_up(learning_rate, init_lr, 
		# # 											global_step_dict[key], **kargs)

		# # 	tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", key, optimizer_type, str(init_lr))
		# # 	opt = self.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

		# # 	if kargs.get("use_tpu", 0) == 1:
		# # 		tf.logging.info("***** Using tpu cross shard optimizer *****")
		# # 		opt = tf.contrib.tpu.CrossShardOptimizer(opt)
		# # 	optimizer_dict[key] = opt

		# switch_acc = tf.get_variable(
		# 					"switch_acc",
		# 					shape=[],
		# 					initializer=tf.constant_initializer(0.0, dtype=tf.float32),
		# 					trainable=False)

		# postive_key = kargs.get("postive_key", "ebm")
		# negative_key = kargs.get("negative_key", "noise")

		# def get_train_op(optimizer, loss, tvars, grad_name):
		# 	grads_and_vars = optimizer_fn.grad_clip_fn(optimizer, loss, tvars, grad_name=postive_key, **kargs)
		# 	with tf.variable_scope(grad_name+"/"+"optimizer", reuse=tf.AUTO_REUSE):
		# 		op = optimizer.apply_gradients(
		# 							grads_and_vars)
		# 	return op

		# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
		# prev_ebm_op = tf.no_op()
		# def ebm_op(prev_ebm_op):
		# 	loop_steps = loop_step_dict[postive_key]

		# 	init_lr = init_lr_dict[postive_key]
		# 	optimizer_type = optimizer_type_dict[postive_key]
		# 	if optimizer_type != 'radam':
		# 		learning_rate = optimizer_fn.private_lr_decay_fn(init_lr, num_train_steps,
		# 												global_step_dict[postive_key], **kargs)
		# 		learning_rate = optimizer_fn.private_warm_up(learning_rate, init_lr, 
		# 											global_step_dict[postive_key], **kargs)

		# 	tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", postive_key, optimizer_type, str(init_lr))
		# 	opt = optimizer_fn.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

		# 	if kargs.get("use_tpu", 0) == 1:
		# 		tf.logging.info("***** Using tpu cross shard optimizer *****")
		# 		opt = tf.contrib.tpu.CrossShardOptimizer(opt)

		# 	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# 	with tf.control_dependencies([update_ops]):
		# 		for i in range(loop_steps):
		# 			with tf.control_dependencies([prev_ebm_op]):
		# 				tvars = tvars_dict[postive_key]
		# 				loop_steps = loop_step_dict[postive_key]
		# 				loss = tf.identity(loss_dict[postive_key])
		# 				prev_ebm_op = get_train_op(opt, loss, tvars, postive_key)
		# 				tf.logging.info("***** model: %s, step: %s *****", postive_key, str(i))
		# 		with tf.control_dependencies([prev_ebm_op]): 
		# 			prev_ebm_op = global_step_dict[postive_key].assign_add(1)
		# 	return prev_ebm_op

		# prev_noise_op = tf.no_op()
		# def noise_op(prev_noise_op):
		# 	loop_steps = loop_step_dict[negative_key]
		# 	init_lr = init_lr_dict[negative_key]
		# 	optimizer_type = optimizer_type_dict[negative_key]
		# 	if optimizer_type != 'radam':
		# 		learning_rate = optimizer_fn.private_lr_decay_fn(init_lr, num_train_steps,
		# 												global_step_dict[negative_key], **kargs)
		# 		learning_rate = optimizer_fn.private_warm_up(learning_rate, init_lr, 
		# 											global_step_dict[negative_key], **kargs)

		# 	tf.logging.info("****** model:%s, optimizer: %s, learning_rate:%s", negative_key, optimizer_type, str(init_lr))
		# 	opt = optimizer_fn.optimizer_op(learning_rate, train_op=optimizer_type, **kargs)

		# 	if kargs.get("use_tpu", 0) == 1:
		# 		tf.logging.info("***** Using tpu cross shard optimizer *****")
		# 		opt = tf.contrib.tpu.CrossShardOptimizer(opt)

		# 	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# 	with tf.control_dependencies([update_ops]):
		# 		for i in range(loop_steps):
		# 			with tf.control_dependencies([prev_noise_op]):
		# 				loss = tf.identity(loss_dict[negative_key])
		# 				tvars = tvars_dict[negative_key]
		# 				loop_steps = loop_step_dict[negative_key]
		# 				prev_noise_op = get_train_op(opt, loss, tvars, negative_key)
		# 				tf.logging.info("***** model: %s, step: %s *****", negative_key, str(i))

		# 		with tf.control_dependencies([prev_noise_op]): 
		# 			prev_noise_op = global_step_dict[negative_key].assign_add(1)
		# 	return prev_noise_op

		# if kargs.get("use_tpu", 0) == 0:
		# 	tf.summary.scalar(postive_key+'_global_step', 
		# 						tf.reduce_sum(global_step_dict[postive_key]))
		# 	tf.summary.scalar(negative_key+'_global_step', 
		# 						tf.reduce_sum(global_step_dict[negative_key]))
		# 	tf.summary.scalar('switch_acc', 
		# 						tf.reduce_sum(switch_acc))

		# prev_op = tf.cond(tf.less_equal(tf.reduce_sum(switch_acc), 0.5),
		# 				   lambda: ebm_op(prev_ebm_op),
		# 				   lambda: noise_op(prev_noise_op))

		# # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# # with tf.control_dependencies(update_ops):
		# with tf.control_dependencies([prev_op]):
		# 	train_op = tf.group(optimizer_fn.global_step.assign_add(1), switch_acc.assign(fce_acc))

		# # return train_op