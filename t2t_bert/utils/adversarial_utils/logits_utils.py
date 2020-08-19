import tensorflow as tf
from utils.bert import bert_utils
from task_module import pretrain, classifier, pretrain_albert

def get_pretrain_logits(model_config,
						model_api, 
						features, 
						labels,
						logits,
						mode,
						target,
						embedding_table_adv=None,
						embedding_seq_adv=None,
						stop_gradient=False,
						sampled_binary_mask=None,
						is_training=True,
						pretrain_loss_type="normal",
						emb_adv_pos="emb_adv_post",
						**kargs):
	model = model_api(model_config, features, labels,
					mode, target, reuse=tf.AUTO_REUSE,
					embedding_table_adv=embedding_table_adv,
					embedding_seq_adv=embedding_seq_adv,
					stop_gradient=stop_gradient,
					emb_adv_pos=emb_adv_pos,
					**kargs)

	if model_config.model_type == 'bert':
		masked_lm_fn = pretrain.get_masked_lm_output
		seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
		print("==apply bert masked lm==")
	elif model_config.model_type == 'albert':
		masked_lm_fn = pretrain_albert.get_masked_lm_output
		seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
		print("==apply albert masked lm==")
	elif model_config.model_type == 'funnelbert':
		masked_lm_fn = pretrain.get_masked_lm_output
		seq_masked_lm_fn = pretrain.seq_mask_masked_lm_output
		print("==apply funnelbert masked lm==")
	else:
		masked_lm_fn = pretrain.get_masked_lm_output
		seq_masked_lm_fn = pretrain_albert.seq_mask_masked_lm_output
		print("==apply bert masked lm==")

	if model_config.get("model_type", "bert") == "funnelbert":
		if n_block > 1 and model_config.get('pretrain_loss', "ae") == "ae":
			seq_masked_lm_fn = pretrain.denoise_autoencoder
			discriminator_mode = model_config.get('discriminator_mode', "ce_loss")
			loss_converage = model_config.get("loss_converage", 'global')
			tf.logging.info("***** discriminator_mode: %s *****"%(discriminator_mode))
			tf.logging.info("***** loss_converage: %s *****"%(loss_converage))
			tf.logging.info(seq_masked_lm_fn)
			model_config.corrupted = True
			tf.logging.info("*** apply reconstruction ***")
			if loss_converage in ['global']:
				sampled_binary_mask = tf.identity(features['input_mask'])
				tf.logging.info("***** loss_converage: %s ***** with input-mask"%(loss_converage))
			elif loss_converage in ['local']:
				sampled_binary_mask = tf.reduce_sum(features['target_mapping'], axis=1)
				tf.logging.info("***** loss_converage: %s ***** with target-mapping mask"%(loss_converage))
		else:
			discriminator_mode = model_config.get('discriminator_mode', "ce_loss")
			loss_converage = model_config.get("loss_converage", 'global')
			tf.logging.info(seq_masked_lm_fn)
	else:
		discriminator_mode = "ce_loss"
		loss_converage = model_config.get("loss_converage", 'global')
		tf.logging.info(seq_masked_lm_fn)
		tf.logging.info(masked_lm_fn)

	if input_ori_ids is not None and model_config.get("corrupted", True):
		(masked_lm_loss,
		masked_lm_example_loss, 
		masked_lm_log_probs,
		masked_lm_mask) = seq_masked_lm_fn(model_config, 
									model.get_sequence_output(output_type=return_type), 
									model.get_embedding_table(),
									features['normal_input_mask'], 
									features['input_ori_ids'], 
									features['input_ids'],
									sampled_binary_mask,
									reuse=tf.AUTO_REUSE,
									embedding_projection=model.get_embedding_projection_table(),
									pretrain_loss_type="normal",
									discriminator_mode=discriminator_mode,
									loss_converage=loss_converage)
		masked_lm_ids = input_ori_ids
		tf.logging.info("*** apply sequential mlm loss ***")
	else:
		masked_lm_positions = features["masked_lm_positions"]
		masked_lm_ids = features["masked_lm_ids"]
		masked_lm_weights = features["masked_lm_weights"]

		(masked_lm_loss,
		masked_lm_example_loss, 
		masked_lm_log_probs,
		masked_lm_mask) = masked_lm_fn(
										model_config, 
										model.get_sequence_output(output_type=return_type), 
										model.get_embedding_table(),
										masked_lm_positions, 
										masked_lm_ids, 
										masked_lm_weights,
										reuse=tf.AUTO_REUSE,
										embedding_projection=model.get_embedding_projection_table(),
										pretrain_loss_type="normal",
										discriminator_mode=discriminator_mode,
										loss_converage=loss_converage)
		tf.logging.info("*** apply bert-like mlm loss ***")

	return masked_lm_log_probs