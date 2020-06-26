from model.bert import bert
from model.bert import bert_rule
from model.bert import bert_seq
from model.bert import albert
from model.bert import bert_electra_joint
from model.bert import albert_official_electra_joint
from model.bert import albert_official
from model.textcnn import textcnn
import tensorflow as tf
from utils.vae import vae_utils
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

	if kargs.get('use_token_type', True):
		tf.logging.info(" use token type ")
	else:
		tf.logging.info(" not use token type ")

	model = bert.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						use_token_type=kargs.get('use_token_type', True),
						reuse=reuse,
						embedding_table_adv=kargs.get('embedding_table_adv', None),
						embedding_seq_adv=kargs.get('embedding_seq_adv', None),
						stop_gradient=kargs.get("stop_gradient", False),
						reuse_mask=kargs.get("reuse_mask", True),
						emb_adv_pos=kargs.get('emb_adv_pos', "emb_adv_post"))
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'),
						reuse_mask=kargs.get("reuse_mask", True))
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
						reuse=reuse,
						embedding_table_adv=kargs.get('embedding_table_adv', None))
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
						input_positions=position_ids,
						embedding_table_adv=kargs.get('embedding_table_adv', None))
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
						input_positions=position_ids,
						embedding_table_adv=kargs.get('embedding_table_adv', None))
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
						input_positions=position_ids,
						embedding_table_adv=kargs.get('embedding_table_adv', None))
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
						input_positions=position_ids,
						embedding_table_adv=kargs.get('embedding_table_adv', None))
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

	print(kargs.get("seq_type", "none"), "===seq type==")
	print(kargs.get("mask_type", "none"), "===mask type==")

	model = bert_seq.Bert(model_config)
	model.build_embedder(input_ids, 
						segment_ids,
						hidden_dropout_prob,
						attention_probs_dropout_prob,
						reuse=reuse,
						past=features.get("past", None),
						decode_loop_step=kargs.get("decode_loop_step", None),
						embedding_table_adv=kargs.get('embedding_table_adv', None))
	model.build_encoder(input_ids,
						input_mask,
						hidden_dropout_prob, 
						attention_probs_dropout_prob,
						reuse=reuse,
						attention_type=kargs.get('attention_type', 'normal_attention'),
						past=features.get("past", None),
						token_type_ids=features.get("segment_ids", None),
						seq_type=kargs.get("seq_type", "none"),
						mask_type=kargs.get("mask_type", "none"),
						decode_loop_step=kargs.get("decode_loop_step", None),
						max_decode_length=kargs.get("max_decode_length", None),
						if_bp=kargs.get("if_bp", False),
						if_cache_decode=kargs.get("if_cache_decode", None))
	model.build_output_logits(reuse=reuse, scope=kargs.get("scope", None))
	# model.build_pooler(reuse=reuse)

	return model

def gated_cnn_encoder(model_config, features, labels, 
			mode, target, reuse=None, **kargs):

	if target:
		input_ids = features["input_ids_{}".format(target)]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)
	else:
		input_ids = features["input_ids"]
		input_char_ids = features.get("input_char_ids_{}".format(target), None)

	if mode == tf.estimator.ModeKeys.TRAIN:
		dropout_prob = model_config.dropout_prob
		is_training = True
	else:
		dropout_prob = 0.0
		is_training = False

	cnn_type = model_config.get("cnn_type", 'dgcnn')

	model = textcnn.TextCNN(model_config)
	model.build_emebdder(input_ids, input_char_ids, is_training, reuse=reuse, **kargs)
	model.build_encoder(input_ids, input_char_ids, is_training, 
						reuse=reuse, 
						cnn_type=cnn_type,
						**kargs)
	if model_config.get('is_casual', True):
		model.build_output_logits(reuse=reuse)
		tf.logging.info(" build seq-lm logits ")
		if cnn_type in ['bi_dgcnn', 'bi_light_dgcnn']:
			tf.logging.info(" build seq-lm-backward logits ")
			model.build_backward_output_logits(reuse=reuse)
	return model

# def gated_cnn_encoder_decoder(model_config, features, labels, 
# 							mode, target, reuse=None, **kargs):

# 	if target:
# 		input_ids = features["input_ids_{}".format(target)]
# 		input_char_ids = features.get("input_char_ids_{}".format(target), None)
# 	else:
# 		input_ids = features["input_ids"]
# 		input_char_ids = features.get("input_char_ids_{}".format(target), None)

# 	if mode == tf.estimator.ModeKeys.TRAIN:
# 		dropout_prob = model_config.dropout_prob
# 		is_training = True
# 	else:
# 		dropout_prob = 0.0
# 		is_training = False

# 	cnn_type = model_config.get("cnn_type", 'dgcnn')
# 	with tf.variable_scope("encoder"):
# 		enc_model = textcnn.TextCNN(model_config)
# 		enc_model.build_emebdder(input_ids, input_char_ids, is_training, 
# 								reuse=reuse, 
# 								**kargs)
# 		enc_model.build_encoder(input_ids, input_char_ids, is_training, 
# 								reuse=reuse, 
# 								cnn_type=cnn_type,
# 								**kargs)
# 		hidden_repres = enc_model.get_pooled_output()

	
	
