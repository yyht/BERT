import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
from data_generator import rule_detector
import json

def full2half(s):
	n = []
	for char in s:
		num = ord(char)
		if num == 0x3000:
			num = 32
		elif 0xFF01 <= num <= 0xFF5E:
			num -= 0xfee0
		num = chr(num)
		n.append(num)
	return ''.join(n)

def get_tokenizer(FLAGS, 
				vocab_path,
				**kargs):
	if FLAGS.tokenizer == "bert":
		tokenizer = tokenization.FullTokenizer(
				vocab_file=vocab_path, 
				do_lower_case=FLAGS.do_lower_case)
	elif FLAGS.tokenizer == "jieba_char":
		tokenizer = tokenization.Jieba_CHAR(
							config=kargs.get("config", {}))

		with tf.gfile.Open(vocab_path, "r") as f:
			lines = f.read().splitlines()
			vocab_lst = []
			for line in lines:
				vocab_lst.append(line)
			print(len(vocab_lst))

		tokenizer.load_vocab(vocab_lst)

	return tokenizer

def get_bert_single_features(FLAGS,
								tokenizer, 
								query,
								max_seq_length):

	tokens_a = tokenizer.tokenize(full2half(query))
	tokens_a = tokens_a[0:max_seq_length-2]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)

	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	feature_dict = {"input_ids":input_ids_a,
			"input_mask":input_mask_a,
			"segment_ids":segment_ids_a,
			"label_ids":[0]}

	return feature_dict

def get_single_features(FLAGS,
							tokenizer, 
							query, 
							max_seq_length):

	tokens_a = tokenizer.tokenize(full2half(query))
			
	if len(tokens_a) > max_seq_length:
		tokens_a = tokens_a[0:(max_seq_length)]

	input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a, max_seq_length)
	if FLAGS.with_char == "char":
		input_char_ids_a = tokenizer.covert_tokens_to_char_ids(tokens_a, 
										max_seq_length, 
										char_len=char_len)
	else:
		input_char_ids_a = None

	feature_dict = {
		"input_ids_a":input_ids_a,
		"label_ids":[0]
	}
	if input_char_ids_a:
		feature_dict["input_char_ids_a"] = input_char_ids_a

	return feature_dict

def get_bert_like_single_features(FLAGS,
								tokenizer, 
								query,
								max_seq_length):

	tokens_a = tokenizer.tokenize((query))
	if len(tokens_a) > max_seq_length:
		tokens_a = tokens_a[0:max_seq_length]

	input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a, max_seq_length)
	if len(input_ids_a) < max_seq_length:
			input_ids_a += [0]*(max_seq_length-len(input_ids_a))

	feature_dict = {
				"input_ids_a":input_ids_a,
				"label_ids":[0]
			}

	return feature_dict

def get_feeddict(FLAGS, vocab_path,
				corpus_path):

	tokenizer_api = get_tokenizer(FLAGS, vocab_path)

	with open(corpus_path, "r") as frobj:

		query_lst = []
		for line in frobj:
			query_lst.append(line.strip())

		instances = []
		for query in query_lst:
			print(query)
			if FLAGS.model_type == "bert":
				feature_dict = get_bert_single_features(
								FLAGS, 
								tokenizer, 
								query, 
								FLAGS.max_seq_length)
			elif FLAGS.model_type == "single_sentence_classification":
				feature_dict = get_single_features(
								FLAGS, 
								tokenizer_api, 
								query, 
								FLAGS.max_seq_length)
			elif FLAGS.model_type == "bert_like_single_sentence_classification":
				feature_dict = get_bert_like_single_features(
								FLAGS, 
								tokenizer_api, 
								query, 
								FLAGS.max_seq_length)
			print(feature_dict)
			instances.append(feature_dict)

	feed_dict = {
		"instances":instances,
		"signature_name":FLAGS.signature_name,
	}

	return feed_dict


