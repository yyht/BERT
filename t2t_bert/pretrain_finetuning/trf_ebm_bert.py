import tensorflow as tf
import numpy as np

from task_module import pretrain, classifier, pretrain_albert
import tensorflow as tf

try:
	from distributed_single_sentence_classification.model_interface import model_zoo
except:
	from distributed_single_sentence_classification.model_interface import model_zoo

from pretrain_finetuning.token_generator import token_generator, random_input_ids_generation
from pretrain_finetuning.token_generator_hmm import hmm_input_ids_generation, ngram_prob

from utils.bert import bert_utils
from model_io import model_io

import copy

def model_fn_builder(
					model_config,
					num_labels,
					init_checkpoint,
					model_reuse=None,
					load_pretrained=True,
					model_io_config={},
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="a",
					**kargs):

	model_config = copy.deepcopy(model_config)
	# if kargs.get("sharing_mode", "none") == "none":
	# 	"""
	# 	'generator/' + model_config.scope
	# 	"""
	# 	model_config.scope = exclude_scope + '/' + model_config.scope
	# 	generator_scope_prefix = exclude_scope
	# 	exclude_scope = exclude_scope
	# 	tf.logging.info("****** generator parameter *******")
	# elif kargs.get("sharing_mode", "none") == "all_sharing":
	# 	generator_scope_prefix = None
	# 	exclude_scope = ''
	# 	tf.logging.info("****** generator parameter sharing with discriminator *******")

	def model_fn(features, labels, mode, params):

		model_api = model_zoo(model_config)

		model = model_api(model_config, features, labels,
							mode, target, reuse=tf.AUTO_REUSE,
							**kargs)

		if mode == tf.estimator.ModeKeys.TRAIN:
			dropout_prob = model_config.dropout_prob
		else:
			dropout_prob = 0.0

		if model_io_config.fix_lm == True:
			scope = model_config.scope + "_finetuning"
		else:
			scope = model_config.scope
		
		logits = pretrain.emb_score(model_config, 
						model.get_sequence_output(), 
						features['input_ids'],
						model.get_embedding_table(),
						features['input_mask'], **kargs)
		
		model_io_fn = model_io.ModelIO(model_io_config)

		pretrained_tvars = model_io_fn.get_params(model_config.scope, 
										not_storage_params=not_storage_params)

		lm_pretrain_tvars = model_io_fn.get_params("cls/predictions", 
									not_storage_params=not_storage_params)

		ebm_pretrain_tvars = model_io_fn.get_params("ebm/predictions", 
									not_storage_params=not_storage_params)

		if model_config.get('embedding_scope', None) is not None:
			embedding_tvars = model_io_fn.get_params(model_config.get('embedding_scope', 'bert')+"/embeddings", 
									not_storage_params=not_storage_params)
			pretrained_tvars.extend(embedding_tvars)

		pretrained_tvars.extend(lm_pretrain_tvars)
		# pretrained_tvars.extend(ebm_pretrain_tvars)
		tvars = pretrained_tvars
		logz_tvars = ebm_pretrain_tvars

		print('==ebm parameters==', tvars)
		print('==ebm logz parameters==', logz_tvars)

		load_vars = tvars + logz_tvars

		if load_pretrained == "yes":
			use_tpu = 1 if kargs.get('use_tpu', False) else 0
			print("==load vars==", load_vars)
			scaffold_fn = model_io_fn.load_pretrained(load_vars, 
											init_checkpoint,
											exclude_scope=exclude_scope,
											use_tpu=use_tpu,
											restore_var_name=model_config.get('restore_var_name', []))
		else:
			scaffold_fn = None

		# logits is logp, when we need to directly maximize it, we only minus
		# with tf.variable_scope("ebm", reuse=tf.AUTO_REUSE):
		# 	ebm_global_step = tf.get_variable(
		# 						"global_step",
		# 						shape=[],
		# 						initializer=tf.constant_initializer(0, dtype=tf.int64),
		# 						trainable=False,
		# 						dtype=tf.int64)
		return_dict = {
					"tvars":tvars,
					"logits":logits,
					"logz_tvars":logz_tvars
					# "global_step":ebm_global_step
				}
		return return_dict
	return model_fn