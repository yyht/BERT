from model.bert import bert
from model.bert import bert_rule
from model.bert import bert_seq
from model.bert import albert
from model.bert import bert_electra_joint
from model.bert import albert_official_electra_joint
from model.bert import albert_official
import tensorflow as tf
import numpy as np

"""
for roberta without nsp prediction, we didn't need
different segment ids, so we just use the same segment ids
as pretraining which is 0-segment ids just like official roberta from 
https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/model.py
that 

self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )

"""

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
	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

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
			mode, target, reuse=None, **kargs):
	
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

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

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

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

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

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

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

def albert_encoder_official(model_config, features, labels, 
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

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = albert_official.Albert(model_config)
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

def electra_gumbel_albert_official_encoder(model_config, features, labels, 
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

	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	model = albert_official_electra_joint.Albert(model_config)
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

def bert_seq_decoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_mask = features["input_mask_{}".format(target)]
		segment_ids = features["segment_ids_{}".format(target)]
	else:
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
	if kargs.get('ues_token_type', 'yes') == 'yes':
		tf.logging.info(" using segment embedding with different types ")
	else:
		tf.logging.info(" using segment embedding with same types ")
		segment_ids = tf.zeros_like(segment_ids)

	if mode == tf.estimator.ModeKeys.TRAIN:
		hidden_dropout_prob = model_config.hidden_dropout_prob
		attention_probs_dropout_prob = 0.0
		dropout_prob = model_config.dropout_prob
	else:
		hidden_dropout_prob = 0.0
		attention_probs_dropout_prob = 0.0
		dropout_prob = 0.0

	tf.logging.info(" hidden_dropout_prob: %s ", str(hidden_dropout_prob))
	tf.logging.info(" attention_probs_dropout_prob: %s ", str(hidden_dropout_prob))
	tf.logging.info(" dropout_prob: %s ", str(dropout_prob))

	model = bert_seq.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse,
						past=features.get("past", None))
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'),
						past=features.get("past", None),
						token_type_ids=features.get("segment_ids", None),
						seq_type=kargs.get("seq_type", "none"),
						mask_type=kargs.get("mask_type", "none"))
	model.build_output_logits(reuse=reuse)
	# model.build_pooler(reuse=reuse)

	return model
	