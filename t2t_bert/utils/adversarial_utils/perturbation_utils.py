import tensorflow as tf
from utils.bert import bert_utils
from utils.adversarial_utils import noise_generator

def adv_perturbation(model_config,
						model_api, 
						features, 
						labels,
						logits,
						mode,
						embedding_table,
						adv_logits_fn,
						embedding_seq_output,
						sampled_binary_mask=None,
						noise_var=1e-5,
						step_size=1e-3,
						noise_gamma=1e-5,
						num_iterations=1,
						is_training=True,
						project_norm_type="l2",
						pretrain_loss_type="normal",
						adv_type="embedding_seq_output",
						stop_gradient=False,
						emb_adv_pos="emb_adv_post",
						adv_method="none",
						noise_type="normal",
						**kargs):

	if noise_type == "normal":
		noise_fn = noise_generator.normal_generation
		tf.logging.info(noise_fn)
	elif noise_type == "uniform":
		noise_fn = noise_generator.uniform_generation
		tf.logging.info(noise_fn)
	else:
		noise_fn = noise_generator.normal_generation
		tf.logging.info(noise_fn)

	if adv_type == 'embedding_table':
		noise = noise_fn(features,
						embedding_table,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding table noise *****")
	elif adv_type == 'embedding_seq_output':
		noise = noise_fn(features,
						embedding_seq_output,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding seq noise *****")
	else:
		noise = noise_fn(features,
						embedding_seq_output,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding seq noise *****")

	tf.logging.info("***** adv hyparameter: noise_var: %s, step_size: %s, noise_gamma: %s" % (str(noise_var), str(step_size), str(noise_gamma)))

	for _ in range(num_iterations):
		if adv_type == 'embedding_table':
			embedding_table_adv = tf.identity(noise)
			embedding_seq_adv = None
			tf.logging.info("***** apply embedding_table *****")
			stop_gradient = False
			emb_adv_pos = "emb_adv_pre"
		elif adv_type == 'embedding_seq_output':
			embedding_table_adv = None
			embedding_seq_adv = tf.identity(noise)
			tf.logging.info("***** apply embedding_seq_output *****")
			emb_adv_pos = "emb_adv_post"

		adv_logits = adv_logits_fn(
									model_config=model_config,
									model_api=model_api, 
									features=features, 
									labels=labels,
									logits=logits,
									mode=mode,
									target=target,
									embedding_table_adv=embedding_table_adv,
									embedding_seq_adv=embedding_seq_adv,
									stop_gradient=stop_gradient,
									sampled_binary_mask=sampled_binary_mask,
									is_training=is_training,
									pretrain_loss_type=pretrain_loss_type, 
									emb_adv_pos=emb_adv_pos,
									**kargs)
		
		dist = kl_divergence_with_logit(tf.stop_gradient(logits), adv_logits)
		if sampled_binary_mask is not None:
			dist = tf.reduce_sum(dist * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))
		else:
			dist = tf.reduce_mean(dist)

		delta_grad = tf.gradients(dist, [noise])[0]
		# add small scale for d update
		delta_grad = tf.stop_gradient(delta_grad)
		
		noise = adv_project(delta_grad, 
						norm_type=project_norm_type, 
						eps=noise_gamma)
		tf.logging.info("***** apply noise proj *****")
		
	if kargs.get('rampup_method', "none") == 'mean_teacher':
		step_size *= tf.random_uniform([])

	return step_size * noise

def rf_perturbation(model_config,
						model_api, 
						features, 
						labels,
						logits,
						mode,
						embedding_table,
						adv_logits_fn,
						embedding_seq_output,
						sampled_binary_mask=None,
						noise_var=1e-5,
						step_size=1e-3,
						noise_gamma=1e-5,
						num_iterations=1,
						is_training=True,
						project_norm_type="l2",
						pretrain_loss_type="normal",
						adv_type="embedding_seq_output",
						stop_gradient=False,
						emb_adv_pos="emb_adv_post",
						adv_method="none",
						noise_type="normal",
						**kargs):
	if noise_type == "normal":
		noise_fn = noise_generator.normal_generation
		tf.logging.info(noise_fn)
	elif noise_type == "uniform":
		noise_fn = noise_generator.uniform_generation
		tf.logging.info(noise_fn)
	else:
		noise_fn = noise_generator.normal_generation
		tf.logging.info(noise_fn)

	if adv_type == 'embedding_table':
		noise = noise_fn(features,
						embedding_table,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding table noise *****")
	elif adv_type == 'embedding_seq_output':
		noise = noise_fn(features,
						embedding_seq_output,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding seq noise *****")
	else:
		noise = noise_fn(features,
						embedding_seq_output,
						noise_var,
						project_norm_type=project_norm_type,
						adv_method=adv_method)
		tf.logging.info("***** apply embedding seq noise *****")

	tf.logging.info("***** adv hyparameter: noise_var: %s, step_size: %s, noise_gamma: %s" % (str(noise_var), str(step_size), str(noise_gamma)))
	return noise
