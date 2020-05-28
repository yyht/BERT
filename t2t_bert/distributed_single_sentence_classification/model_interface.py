from distributed_encoder.bert_encoder import bert_encoder
from distributed_encoder.bert_encoder import bert_rule_encoder
from distributed_encoder.bert_encoder import bert_seq_decoder
from distributed_encoder.gpt_encoder import gpt_encoder
from distributed_encoder.bert_encoder import albert_encoder
from distributed_encoder.bert_encoder import electra_gumbel_encoder
from distributed_encoder.bert_encoder import albert_encoder_official
from distributed_encoder.bert_encoder import electra_gumbel_albert_official_encoder
from distributed_encoder.bert_encoder import gated_cnn_encoder

from distributed_encoder.classifynet_encoder import textcnn_encoder
from distributed_encoder.classifynet_encoder import textlstm_encoder
from distributed_encoder.interaction_encoder import match_pyramid_encoder
from distributed_encoder.interaction_encoder import textcnn_interaction_encoder
from distributed_encoder.interaction_encoder import bert_interaction_encoder
from distributed_encoder.classifynet_encoder import dan_encoder

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
	elif model_config.get("model_type", "bert") in ["textcnn", "textcnn_distillation", 
													"textcnn_distillation_adv_adaptation"]:
		print("==apply textcnn encoder==")
		model_interface = textcnn_encoder
	elif model_config.get("model_type", "bert_small") == "bert_small":
		print("==apply bert small encoder==")
		model_interface = bert_encoder
	elif model_config.get("model_type", "bert") in ["textlstm", "textlstm_distillation"]:
		model_interface = textlstm_encoder
	elif model_config.get("model_type", "match_pyramid") in ["match_pyramid", "match_pyramid_distillation"]:
		model_interface = match_pyramid_encoder
	elif model_config.get("model_type", "textcnn_interaction") in ["textcnn_interaction"]:
		model_interface = textcnn_interaction_encoder
	elif model_config.get("model_type", "match_pyramid") in ["dan", "dan_distillation"]:
		model_interface = dan_encoder
	elif model_config.get('model_type', 'gpt') in ['gpt']:
		model_interface = gpt_encoder
	elif model_config.get("model_type", "albert") == "albert": 
		model_interface = albert_encoder
	elif model_config.get("model_type", "electra_gumbel_encoder") == "electra_gumbel_encoder":
		model_interface = electra_gumbel_encoder
	elif  model_config.get("model_type", "albert_official") == "albert_official":
		model_interface = albert_encoder_official
	elif model_config.get("model_type", "electra_gumbel_albert_official_encoder") == "electra_gumbel_albert_official_encoder":
		model_interface = electra_gumbel_albert_official_encoder
	elif model_config.get("model_type", "bert_seq") == "bert_seq":
		model_interface = bert_seq_decoder
		tf.logging.info("****** bert seq encoder ******* ")
	elif model_config.get("model_type", "bert") == "gated_cnn_seq":
		model_interface = gated_cnn_encoder
		tf.logging.info("****** bert seq encoder ******* ")
	elif model_config.get("model_type", "bert") == "bert_interaction":
		model_interface = bert_interaction_encoder
		tf.logging.info("****** bert seq encoder ******* ")
	return model_interface

def model_config_parser(FLAGS):

	print(FLAGS.model_type)

	if FLAGS.model_type in ["bert", "bert_rule", "albert", "electra_gumbel_encoder", 
					"albert_official", "electra_gumbel_albert_official_encoder",
					"bert_seq", "bert_interaction"]:
		config = json.load(open(FLAGS.config_file, "r"))
		print(config, '==model config==')
		config = Bunch(config)
		config.use_one_hot_embeddings = True
		# if FLAGS.exclude_scope:
		#	config.scope = FLAGS.exclude_scope + "/" + "bert"
		#	tf.logging.info("****** add exclude_scope ******* %s", str(config.scope))
	#	else:
		# 	config.scope = FLAGS.exclude_scope + "/" + "bert"
		# 	tf.logging.info("****** add exclude_scope ******* %s", str(config.scope))
		# else:
		config.scope = FLAGS.model_scope #"bert"
		tf.logging.info("****** original scope ******* %s", str(config.scope))
		config.dropout_prob = 0.1
		try:
			config.label_type = FLAGS.label_type
		except:
			config.label_type = "single_label"
		tf.logging.info("****** label type ******* %s", str(config.label_type))
		config.model_type = FLAGS.model_type
		config.ln_type = FLAGS.ln_type
		if FLAGS.task_type in ['bert_pretrain']:
			if FLAGS.load_pretrained == "yes":
				config.init_lr = FLAGS.init_lr
				config.warmup = 0.1
			else:
				config.init_lr = FLAGS.init_lr
				config.warmup = 0.1
			print('==apply bert pretrain==', config.init_lr)
		else:
			if FLAGS.model_type in ['albert']:
				try:
					config.init_lr = FLAGS.init_lr
				except:
					config.init_lr = 1e-4
			else:
				# try:
				print(FLAGS)
				config.init_lr = FLAGS.init_lr
				# except:
				# 	config.init_lr = 2e-5
			print('==apply albert finetuning==', config.init_lr)
		print("===learning rate===", config.init_lr)
		try:
			if FLAGS.attention_type in ['rezero_transformer']:
				config.warmup = 0.0
				tf.logging.info("****** warmup ******* %s", str(config.warmup))
		except:
			tf.logging.info("****** normal attention ******* ")
		tf.logging.info("****** learning rate ******* %s", str(config.init_lr))
		# config.loss = "dmi_loss"

		try:
			config.loss = FLAGS.loss
		except:
			config.loss = "entropy"
		tf.logging.info("****** loss type ******* %s", str(config.loss))

		# config.loss = "focal_loss"
		config.rule_type_size = 2
		config.lm_ratio = 1.0
		config.max_length = FLAGS.max_length
		config.nsp_ratio = 1.0
		config.max_predictions_per_seq = FLAGS.max_predictions_per_seq
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
		config.init_lr = 3e-5
		config.num_hidden_layers = FLAGS.num_hidden_layers
		config.loss = "entropy"
		config.rule_type_size = 2
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["textcnn", 'textcnn_distillation', 
								'textcnn_distillation_adv_adaptation',
								'textcnn_interaction']:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)
		config = json.load(open(FLAGS.config_file, "r"))

		[w2v_embed, token2id, 
		id2token, is_extral_symbol, use_pretrained] = load_w2v.load_pretrained_w2v(vocab_path, w2v_path, config.get('emb_size', 64))
		
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
		config.use_pretrained = use_pretrained
		config.label_type = FLAGS.label_type
		if is_extral_symbol == 1:
			config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
			print("==need extra_symbol==")

		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["textlstm", "textlstm_distillation"]:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		[w2v_embed, token2id, 
		id2token, is_extral_symbol] = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
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
		if is_extral_symbol == 1:
			config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
			print("==need extra_symbol==")
		
		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ["match_pyramid", "match_pyramid_distillation"]:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		[w2v_embed, token2id, 
		id2token, is_extral_symbol] = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
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
		if is_extral_symbol == 1:
			config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
			print("==need extra_symbol==")
		config.max_seq_len = FLAGS.max_length
		if FLAGS.task_type in ["interaction_pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

		if config.compress_emb:
			config.embedding_dim_compressed = config.cnn_num_filters

	elif FLAGS.model_type in ["dan", 'dan_distillation']:
		from data_generator import load_w2v
		w2v_path = os.path.join(FLAGS.buckets, FLAGS.w2v_path)
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)

		print(w2v_path, vocab_path)

		[w2v_embed, token2id, 
		id2token, is_extral_symbol] = load_w2v.load_pretrained_w2v(vocab_path, w2v_path)
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.token_emb_mat = w2v_embed
		config.char_emb_mat = None
		config.vocab_size = w2v_embed.shape[0]
		config.max_length = FLAGS.max_length
		config.emb_size = w2v_embed.shape[1]
		config.scope = "dan"
		config.char_dim = w2v_embed.shape[1]
		config.char_vocab_size = w2v_embed.shape[0]
		config.char_embedding = None
		config.model_type = FLAGS.model_type
		config.dropout_prob = config.dropout_rate
		config.init_lr = config.learning_rate
		if is_extral_symbol == 1:
			config.extra_symbol = ["<pad>", "<unk>", "<s>", "</s>"]
			print("==need extra_symbol==")

		if FLAGS.task_type in ["pair_sentence_classification"]:
			config.classifier = FLAGS.classifier
			config.output_layer = FLAGS.output_layer

	elif FLAGS.model_type in ['gpt']:
		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.dropout_prob = 0.1
		config.init_lr = 1e-4

	elif FLAGS.model_type in ["gated_cnn_seq"]:

		config = json.load(open(FLAGS.config_file, "r"))
		config = Bunch(config)
		config.token_emb_mat = None
		config.char_emb_mat = None
		config.vocab_size = config.vocab_size
		config.max_length = FLAGS.max_length
		config.emb_size = config.emb_size
		config.scope = "textcnn"
		config.char_dim = config.emb_char_size
		config.char_vocab_size = config.vocab_size
		config.char_embedding = None
		config.model_type = FLAGS.model_type
		config.dropout_prob = config.dropout_rate
		config.init_lr = FLAGS.init_lr
		config.grad_clip = "gloabl_norm"
		config.clip_norm = 10.0
		config.max_seq_len = FLAGS.max_length

	return config
