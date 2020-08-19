import tensorflow as tf
from utils.bert import bert_utils
from utils.adversarial_utils import logits_utils
from utils.adversarial_utils import perturbation_utils

"""
taken from https://github.com/takerum/vat_tf
"""

def kl_divergence_with_logit(q_logit, p_logit):
	# [batch_size, seq_length, classes]
	q_logit = tf.nn.log_softmax(q_logit, axis=-1)
	p_logit = tf.nn.log_softmax(p_logit, axis=-1)

	# [batch_size, seq_length]
	qlogq = tf.reduce_sum(tf.exp(q_logit) * q_logit, -1)
	qlogp = tf.reduce_sum(tf.exp(q_logit) * p_logit, -1)
	return qlogq - qlogp

def adv_project(grad, norm_type='inf', eps=1e-6):
	"""
	taken from 
	https://github.com/namisan/mt-dnn/blob/master/alum/adv_masked_lm.py
	input_shape = bert_utils.get_shape_list(d)

	def adv_project(self, grad, norm_type='inf', eps=1e-6):
				if norm_type == 'l2':
						direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
				elif norm_type == 'l1':
						direction = grad.sign()
				else:
						direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
				return direction

	"""
	input_shape = bert_utils.get_shape_list(grad)
	if len(input_shape) >= 3:
		shape_list = range(1, len(input_shape))
	else:
		shape_list = range(0, len(input_shape))
	if norm_type == 'l2':
		alpha = tf.reduce_max(tf.abs(grad), shape_list, keep_dims=True) + 1e-12
		l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(grad / alpha, 2), shape_list, keep_dims=True) + eps)
		direction = grad / l2_norm
		tf.logging.info("***** apply l2-adv *****")
	elif norm_type == 'l1':
		direction = tf.sign(grad)
		tf.logging.info("***** apply l1-adv *****")
	else:
		grad_max = tf.reduce_max(tf.abs(grad), 
								axis=shape_list,
								keep_dims=True)
		direction = grad / (grad_max + eps)
		tf.logging.info("***** apply inf-adv *****")
	return direction

def adversarial_loss(model_config,
							model_api, 
							features, 
							labels,
							logits,
							mode,
							target,
							embedding_table,
							adv_logits_fn,
							perturbation_fn,
							embedding_seq_output,
							sampled_binary_mask=None,
							num_iterations=1,
							noise_var=1e-5,
							step_size=1e-3,
							noise_gamma=1e-5,
							is_training=True,
							project_norm_type="l2",
							pretrain_loss_type="normal",
							adv_type="embedding_seq_output",
							stop_gradient=False,
							emb_adv_pos="emb_adv_post",
							perturbation_type="adv_perturbation",
							adv_method="none",
							noise_type="normal",
							**kargs):
	"""
	https://github.com/namisan/mt-dnn/blob/master/alum/adv_masked_lm.py
	"""
	if perturbation_type == "adv_perturbation":
		perturbation_fn = perturbation_utils.adv_perturbation
	elif perturbation_type == "rf_perturbation":
		perturbation_fn = perturbation_utils.rf_perturbation
	else:
		perturbation_fn = None
	if perturbation_fn:
		r_vadv = perturbation_fn(
								model_config=model_config,
								model_api=model_api, 
								features=features, 
								labels=labels,
								logits=logits,
								mode=mode,
								target=target,
								embedding_table=embedding_table,
								adv_logits_fn=adv_logits_fn,
								embedding_seq_output=embedding_seq_output,
								stop_gradient=stop_gradient,
								sampled_binary_mask=sampled_binary_mask,
								noise_var=noise_var,
								step_size=step_size,
								noise_gamma=noise_gamma,
								num_iterations=num_iterations,
								is_training=is_training,
								project_norm_type=project_norm_type,
								pretrain_loss_type=pretrain_loss_type,
								adv_type=adv_type,
								emb_adv_pos=emb_adv_pos,
								adv_method=adv_method,
								noise_type=noise_type,
								**kargs)

		if adv_type == 'embedding_table':
			if kargs.get("no_embed_bp_adv", False):
				embedding_table_adv = r_vadv
				tf.logging.info("***** embed_bp_adv *****")
			else:
				embedding_table_adv = r_vadv+tf.stop_gradient(embedding_table)-embedding_table
				tf.logging.info("***** no_embed_bp_adv *****")
			tf.logging.info("***** apply embedding_table *****")
			embedding_seq_adv = None
		elif adv_type == 'embedding_seq_output':
			embedding_table_adv = None
			embedding_seq_adv = r_vadv 
			tf.logging.info("***** apply embedding_seq_output *****")

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
									**kargs)

		dist_b = kl_divergence_with_logit(tf.stop_gradient(logits), adv_logits)
		if sampled_binary_mask is not None:
			dist_b = tf.reduce_sum(dist_b * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))

		if kargs.get("kl_inclusive", False):
			dist_f = kl_divergence_with_logit(tf.stop_gradient(adv_logits), logits)
			if sampled_binary_mask is not None:
				dist_f = tf.reduce_sum(dist_f * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))
			loss = tf.reduce_sum(dist_b+dist_f)/2
			tf.logging.info("***** apply kl_inclusive *****")
		else:
			loss = tf.reduce_sum(dist_b)
			tf.logging.info("***** apply kl_exclusive *****")
		return tf.identity(loss, name='adv_loss')
	else:
		return tf.constant(0.0)
