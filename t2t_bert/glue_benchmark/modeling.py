from model.bert import bert
from model.bert import bert_rule
from model.bert import albert
from model.bert import bert_electra_joint
import tensorflow as tf
import numpy as np
from bunch import Bunch
from utils.bert import bert_utils
import copy
from model_io import model_io_utils

def BertModel(bert_config, is_training, input_ids, input_mask, segment_ids,
				use_one_hot_embeddings,
				**kargs):

	config = copy.deepcopy(bert_config)
	model_config = Bunch(config)
	model_config.use_one_hot_embeddings = use_one_hot_embeddings

	if kargs.get('exclude_scope', None):
		model_config.scope = exclude_scope + '/' + 'bert'
	else:
		model_config.scope = 'bert'

	model_config.ln_type = kargs.get('ln_type', 'postln')
	tf.logging.info(" ln type %s ", model_config.ln_type)

	if not is_training:
		model_config.hidden_dropout_prob = 0.0
		model_config.attention_probs_dropout_prob = 0.0

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

	if kargs.get('model_type', 'bert') == 'bert':
		model_fn = bert.Bert
	elif kargs.get('model_type', 'albert') == 'albert':
		model_fn = albert.Bert
	elif kargs.get('model_type', 'electra') == 'electra':
		model_fn = bert_electra_joint.Bert
	else:
		model_fn = bert.Bert

	model = model_fn(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						model_config.hidden_dropout_prob,
						model_config.attention_probs_dropout_prob,
						reuse=tf.AUTO_REUSE)
	model.build_encoder(input_ids,
						input_mask,
						model_config.hidden_dropout_prob, 
						model_config.attention_probs_dropout_prob,
						reuse=tf.AUTO_REUSE,
						attention_type=kargs.get('attention_type', 'normal_attention'))
	model.build_pooler(reuse=tf.AUTO_REUSE)

	return model

def get_shape_list(tensor, expected_rank=None, name=None):
	return bert_utils.get_shape_list(tensor, expected_rank, name)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint, **kargs):
	[
		assignment_map, 
		initialized_variable_names
    ] = model_io_utils.get_assigment_map_from_checkpoint(tvars, init_checkpoint, **kargs)

	return [assignment_map,
			initialized_variable_names]
