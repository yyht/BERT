from distributed_encoder.bert_encoder import bert_encoder
from distributed_encoder.bert_encoder import bert_rule_encoder

from distributed_encoder.classifynet_encoder import textcnn_encoder
from distributed_encoder.classifynet_encoder import textlstm_encoder
from distributed_encoder.interaction_encoder import match_pyramid_encoder

import tensorflow as tf
import numpy as np
import json
from bunch import Bunch
import os, sys


def model_zoo(model_config):
	if model_config.get("model_type", "bert") == "bert":
		print("==apply bert encoder==")
		model_interface = bert_encoder
	elif model_config.get("model_type", "bert") == "bert_rule":
		print("==apply bert rule encoder==")
		model_interface = bert_rule_encoder
	elif model_config.get("model_type", "bert") in ["textcnn", "textcnn_distillation"]:
		print("==apply textcnn encoder==")
		model_interface = textcnn_encoder
	elif model_config.get("model_type", "bert_small") == "bert_small":
		print("==apply bert small encoder==")
		model_interface = bert_encoder
	elif model_config.get("model_type", "bert") in ["textlstm", "textlstm_distillation"]:
		model_interface = textlstm_encoder
	elif model_config.get("model_type", "match_pyramid") in ["match_pyramid", "match_pyramid_distillation"]:
		model_interface = match_pyramid_encoder

	return model_interface

def model_config_parser(FLAGS):

	print(FLAGS.model_type)

	if FLAGS.model_type in ["bert", "bert_rule"]:
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		config.model_type = FLAGS.model_type
		config.init_lr = 2e-5
		config.loss = "entropy"
		config.rule_type_size = 2
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier

	elif FLAGS.model_type in ["bert_small"]:
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		config.model_type = FLAGS.model_type
		config.init_lr = 2e-5
		config.num_hidden_layers = FLAGS.num_hidden_layers
		config.loss = "entropy"
		config.rule_type_size = 2
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["textcnn", 'textcnn_distillation']:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		w2v_embed, token2id, id2token = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.token_emb_mat = w2v_embed
		config.char_emb_mat = None
		config.vocab_size = w2v_embed.shape[0]
		config.max_length = FLAGS.max_length
		config.emb_size = w2v_embed.shape[1]
		config.scope = "textcnn"
		config.char_dim = w2v_embed.shape[1]
		config.char_vocab_size = w2v_embed.shape[0]
		config.char_embedding = None
		config.model_type = FLAGS.model_type
		config.dropout_prob = config.dropout_rate
		config.init_lr = config.learning_rate
		# config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
		
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["textlstm", "textlstm_distillation"]:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		w2v_embed, token2id, id2token = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.token_emb_mat = w2v_embed
		config.char_emb_mat = None
		config.vocab_size = w2v_embed.shape[0]
		config.max_length = FLAGS.max_length
		config.emb_size = w2v_embed.shape[1]
		config.scope = "textlstm"
		config.char_dim = w2v_embed.shape[1]
		config.char_vocab_size = w2v_embed.shape[0]
		config.char_embedding = None
		config.model_type = FLAGS.model_type
		config.dropout_prob = config.dropout_rate
		config.init_lr = config.learning_rate
		config.grad_clip = "gloabl_norm"
		config.clip_norm = 5.0
		config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["match_pyramid", "match_pyramid_distillation"]:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		w2v_embed, token2id, id2token = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.token_emb_mat = w2v_embed
		config.char_emb_mat = None
		config.vocab_size = w2v_embed.shape[0]
		config.max_length = FLAGS.max_length
		config.emb_size = w2v_embed.shape[1]
		config.scope = "match_pyramid"
		config.char_dim = w2v_embed.shape[1]
		config.char_vocab_size = w2v_embed.shape[0]
		config.char_embedding = None
		config.model_type = FLAGS.model_type
		config.dropout_prob = config.dropout_rate
		config.init_lr = config.learning_rate
		config.grad_clip = "gloabl_norm"
		config.clip_norm = 5.0
		config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
		config.max_seq_len = FLAGS.max_length
		if FLAGS.task_type in ["interaction_pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

		if config.compress_emb:
			config.embedding_dim_compressed = config.cnn_num_filters

	return config