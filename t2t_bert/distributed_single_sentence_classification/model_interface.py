from distributed_encoder.bert_encoder import bert_encoder
from distributed_encoder.bert_encoder import bert_rule_encoder

# from distributed_encoder.classifynet_encoder import textcnn_encoder

import tensorflow as tf
import numpy as np
import json

# model_zoo = {
# 	"bert":bert_encoder,
# 	"bert_rule":bert_rule_encoder
# }

def model_zoo(model_config):
	if model_config.get("model_type", "bert") == "bert":
		print("==apply bert encoder==")
		model_interface = bert_encoder
	elif model_config.get("model_type", "bert") == "bert_rule":
		print("==apply bert rule encoder==")
		model_interface = bert_rule_encoder
	# elif model_config.get("model", "bert") == "textcnn":
	# 	model_interface = textcnn_encoder

	return model_interface

def model_config_parser(FLAGS):

	if FLAGS.model_type in ["bert", "bert_rule"]:
		config = json.load(open(FLAGS.config_file, "r"))

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		config.model = FLAGS.model_type

	elif FLAGS.model_type == "textcnn":
		from data_generator import load_w2v
		w2v_embed, token2id, id2token = load_pretrained_w2v(FLAGS.w2v_path)
		config = Bunch({})
		config.token_emb_mat = w2v_embed
		config.char_emb_mat = None
		config.vocab_size = len(token2id)
		config.max_length = FLAGS.max_length
		config.emb_size = w2v_embed.shape[1]
		config.scope = "textcnn"
		config.char_dim = w2v_embed.shape[1]

		config.model = FLAGS.model_type

	return config