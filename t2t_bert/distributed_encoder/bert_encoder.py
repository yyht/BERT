from model.bert import bert
from model.bert import bert_rule
from model.bert import albert
from model.bert import bert_electra_joint
import tensorflow as tf
import numpy as np

def bert_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_mask = features["input_mask_{}".format(target)]
		segment_ids = features["segment_ids_{}".format(target)]
	else:
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = bert.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'))
	model.build_pooler(reuse=reuse)

	return model

def bert_rule_encoder(model_config, features, labels, 
			mode, target, reuse=None):
	
	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_mask = features["input_mask_{}".format(target)]
		segment_ids = features["segment_ids_{}".format(target)]
		rule_ids = features["rule_ids_{}".format(target)]
	else:
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		rule_ids = features["rule_ids"]

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = bert_rule.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						rule_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_pooler(reuse=reuse)

	return model

def albert_encoder(model_config, features, labels, 
			mode, target, reuse=None):

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_mask = features["input_mask_{}".format(target)]
		segment_ids = features["segment_ids_{}".format(target)]
		position_ids = features.get("position_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		position_ids = features.get("position_ids".format(target), None)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = albert.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse,
						input_positions=position_ids)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse)
	model.build_pooler(reuse=reuse)

	return model

def electra_gumbel_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):
	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_mask = features["input_mask_{}".format(target)]
		segment_ids = features["segment_ids_{}".format(target)]
		position_ids = features.get("position_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		position_ids = features.get("position_ids".format(target), None)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = bert_electra_joint.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse,
						input_positions=position_ids)
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'))
	model.build_pooler(reuse=reuse)

	return model
	