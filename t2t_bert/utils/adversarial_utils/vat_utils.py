import tensorflow as tf
from utils.bert import bert_utils
from task_module.pretrain import get_masked_lm_output, seq_mask_masked_lm_output
from task_module import pretrain, classifier, pretrain_albert

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

# def get_normalized_vector(d):
# 	input_shape = bert_utils.get_shape_list(d)
# 	d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(input_shape)), keep_dims=True))
# 	d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(input_shape)), keep_dims=True))
# 	return d

def get_pretrain_logits(model_config,
						model_api, 
						features, 
						labels,
						logits,
						mode,
						target,
						embedding_table_adv=None,
						sampled_binary_mask=None,
						is_training=True,
						pretrain_loss_type="normal", 
						**kargs):
	model = model_api(model_config, features, labels,
					mode, target, reuse=tf.AUTO_REUSE,
					embedding_table_adv=embedding_table_adv,
					**kargs)

	if model_config.model_type == 'bert':
		masked_lm_fn = pretrain.get_masked_lm_output
		seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
		print("==apply bert masked lm==")
	elif model_config.model_type == 'albert':
		masked_lm_fn = pretrain_albert.get_masked_lm_output
		seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
		print("==apply albert masked lm==")
	else:
		masked_lm_fn = pretrain.get_masked_lm_output
		seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
		print("==apply bert masked lm==")

	if sampled_binary_mask is not None:
		(masked_lm_loss,
		masked_lm_example_loss, 
		masked_lm_log_probs,
		masked_lm_mask) = seq_masked_lm_fn(model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(),
									features['input_mask'], 
									features['input_ori_ids'], 
									features['input_ids'],
									sampled_binary_mask,
									reuse=tf.AUTO_REUSE,
									embedding_projection=model.get_embedding_projection_table(),
									pretrain_loss_type=pretrain_loss_type)
	else:
		masked_lm_positions = features["masked_lm_positions"]
		masked_lm_ids = features["masked_lm_ids"]
		masked_lm_weights = features["masked_lm_weights"]

		(masked_lm_loss,
		masked_lm_example_loss, 
		masked_lm_log_probs,
		masked_lm_mask) = masked_lm_fn(
									model_config, 
									model.get_sequence_output(), 
									model.get_embedding_table(),
									masked_lm_positions, 
									masked_lm_ids, 
									masked_lm_weights,
									reuse=tf.AUTO_REUSE,
									embedding_projection=model.get_embedding_projection_table(),
									pretrain_loss_type=pretrain_loss_type)
	return masked_lm_log_probs

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
	if norm_type == 'l2':
		grad_norm = tf.norm(grad, 
							axis=range(1, len(input_shape)),
							keep_dims=True)
		direction = grad / (grad_norm + eps)
	elif norm_type == 'l1':
		direction = tf.sign(grad)
	else:
		grad_max = tf.reduce_max(tf.abs(grad), 
								axis=range(1, len(input_shape)),
								keep_dims=True)
		direction = grad / (grad_max + eps)
	return direction

def generate_virtual_adversarial_perturbation(model_config,
											model_api, 
											features, 
											labels,
											logits,
											mode,
											target,
											sampled_binary_mask=None,
											noise_var=1e-5,
											step_size=1e-5,
											noise_gamma=1e-6,
											num_power_iterations=1,
											is_training=True,
											project_norm_type="l2", 
											**kargs):
	input_shape = bert_utils.get_shape_list(model.get_embedding_table())
	noise = tf.random_normal(shape=input_shape) * noise_var

	for _ in range(num_power_iterations):
		adv_logits = get_pretrain_logits(
									model_config=model_config,
									model_api=model_api, 
									features=features, 
									labels=labels,
									logits=logits,
									mode=mode,
									target=target,
									embedding_table_adv=noise,
									sampled_binary_mask=sampled_binary_mask,
									is_training=is_training,
									**kargs)
		
		dist = kl_divergence_with_logit(logits, adv_logits)
		if sampled_binary_mask:
			dist = tf.reduce_sum(dist * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))
		else:
			dist = tf.reduce_mean(dist)
			
		delta_grad = tf.gradients(dist, [noise], aggregation_method=2)[0]
		# add small scale for d update
		delta_grad = tf.stop_gradient(delta_grad)
		noise = adv_project(delta_grad, 
						norm_type=project_norm_type, 
						eps=noise_gamma)

	return step_size * noise

def virtual_adversarial_loss(model_config,
							model_api, 
							features, 
							labels,
							logits,
							mode,
							target,
							sampled_binary_mask=None,
							num_power_iterations=1,
							noise_var=1e-5,
							step_size=1e-5,
							noise_gamma=1e-6,
							is_training=True,
							project_norm_type="l2",
							**kargs):
	"""
	https://github.com/namisan/mt-dnn/blob/master/alum/adv_masked_lm.py
	"""
	r_vadv = generate_virtual_adversarial_perturbation(
								model_config=model_config,
								model_api=model_api, 
								features=features, 
								labels=labels,
								logits=logits,
								mode=mode,
								target=target,
								sampled_binary_mask=sampled_binary_mask,
								noise_var=noise_var,
								step_size=step_size,
								noise_gamma=noise_gamma,
								num_power_iterations=num_power_iterations,
								is_training=is_training,
								project_norm_type=project_norm_type,
								**kargs)

	adv_logits = get_pretrain_logits(
								model_config=model_config,
								model_api=model_api, 
								features=features, 
								labels=labels,
								logits=logits,
								mode=mode,
								target=target,
								embedding_table_adv=r_vadv,
								sampled_binary_mask=sampled_binary_mask,
								is_training=is_training, 
								**kargs)

	dist_b = kl_divergence_with_logit(tf.stop_gradients(logits), adv_logits)
	dist_f = kl_divergence_with_logit(tf.stop_gradients(adv_logits), logits)
	if sampled_binary_mask:
		dist_b = tf.reduce_sum(dist_b * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))
		dist_f = tf.reduce_sum(dist_f * tf.cast(sampled_binary_mask, tf.float32)) / tf.reduce_sum(1e-10+tf.cast(sampled_binary_mask, tf.float32))
		
	loss = tf.reduce_mean(dist_b+dist_f)
	return tf.identity(loss, name='vat_loss')