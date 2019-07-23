import sys,os,json

# -*- coding: utf-8 -*-
import sys,os

father_path = os.path.join(os.getcwd())
print(father_path, "==father path==")

def find_bert(father_path):
	if father_path.split("/")[-1] == "BERT":
		return father_path

	output_path = ""
	for fi in os.listdir(father_path):
		if fi == "BERT":
			output_path = os.path.join(father_path, fi)
			break
		else:
			if os.path.isdir(os.path.join(father_path, fi)):
				find_bert(os.path.join(father_path, fi))
			else:
				continue
	return output_path

bert_path = find_bert(father_path)
t2t_bert_path = os.path.join(bert_path, "t2t_bert")
sys.path.extend([bert_path, t2t_bert_path])

import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
import json

from example import feature_writer, write_to_tfrecords
from example import classifier_processor
from data_generator import vocab_filter

flags = tf.flags

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"test_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"dev_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"train_result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"test_result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"dev_result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_bool(
	"lower_case", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"if_rule", "rule",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"rule_word_path", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"rule_word_dict", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"rule_label_dict", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"with_char", "char",
	"if apply rule detector"
	)

flags.DEFINE_integer(
	"char_len", 10,
	"if apply rule detector"
	)

flags.DEFINE_string(
	"config", "",
	"if apply rule detector"
	)

flags.DEFINE_integer(
	"predefined_vocab_size", 10000,
	"if apply rule detector"
	)

flags.DEFINE_string(
	"corpus_vocab_path", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"data_type", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"tokenizer_type", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"label_type", "multi_class",
	"if apply rule detector"
	)

def main(_):

	# tokenizer = tokenization.Jieba_CHAR(
	# 	config=FLAGS.config)

	# with tf.gfile.Open(FLAGS.vocab_file, "r") as f:
	# 	vocab_lst = []
	# 	for line in f:
	# 		vocab_lst.append(line.strip())

	# vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_file)
	vocab_path = FLAGS.vocab_file
	train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	test_file = os.path.join(FLAGS.buckets, FLAGS.test_file)
	dev_file = os.path.join(FLAGS.buckets, FLAGS.dev_file)

	train_result_file = os.path.join(FLAGS.buckets, FLAGS.train_result_file)
	test_result_file = os.path.join(FLAGS.buckets, FLAGS.test_result_file)
	dev_result_file = os.path.join(FLAGS.buckets, FLAGS.dev_result_file)

	corpus_vocab_path = os.path.join(FLAGS.buckets, FLAGS.corpus_vocab_path)

	if FLAGS.tokenizer_type == "jieba":
		tokenizer = tokenization.Jieba_CHAR(
			config=FLAGS.config)
	elif FLAGS.tokenizer_type == "full_bpe":
		tokenizer = tokenization.FullTokenizer(
				vocab_file=vocab_path, 
				do_lower_case=FLAGS.lower_case)

	if FLAGS.tokenizer_type == "jieba":
		print(FLAGS.with_char)
		with tf.gfile.Open(vocab_path, "r") as f:
			lines = f.read().splitlines()
			vocab_lst = []
			for line in lines:
				vocab_lst.append(line)
			print(len(vocab_lst))

		tokenizer.load_vocab(vocab_lst)

	print("==not apply rule==")
	if FLAGS.data_type == "fasttext":
		classifier_data_api = classifier_processor.FasttextClassifierProcessor()

	classifier_data_api.get_labels(FLAGS.label_id)

	train_examples = classifier_data_api.get_train_examples(train_file,
										is_shuffle=True)
	print("==total train examples==", len(train_examples))

	test_examples = classifier_data_api.get_train_examples(test_file,
										is_shuffle=False)
	print("==total test examples==", len(test_examples))

	dev_examples = classifier_data_api.get_train_examples(dev_file,
										is_shuffle=False)
	print("==total dev examples==", len(dev_examples))


	if FLAGS.tokenizer_type == "jieba":
		vocab_filter.vocab_filter(train_examples+test_examples+dev_examples, vocab_lst, 
								tokenizer, FLAGS.predefined_vocab_size, 
								corpus_vocab_path)

		tokenizer_corpus = tokenization.Jieba_CHAR(
			config=FLAGS.config)

		with tf.gfile.Open(corpus_vocab_path, "r") as f:
			lines = f.read().splitlines()
			vocab_lst = []
			for line in lines:
				vocab_lst.append(line)
			print(len(vocab_lst))
			# print(vocab_lst)

		tokenizer_corpus.load_vocab(vocab_lst)
	elif FLAGS.tokenizer_type == "full_bpe":
		tokenizer_corpus = tokenizer

	if FLAGS.tokenizer_type == "jieba":
		write_to_tfrecords.convert_distillation_classifier_examples_to_features(train_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																train_result_file,
																FLAGS.with_char,
																FLAGS.char_len)

		write_to_tfrecords.convert_distillation_classifier_examples_to_features(test_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																test_result_file,
																FLAGS.with_char,
																FLAGS.char_len)

		write_to_tfrecords.convert_distillation_classifier_examples_to_features(dev_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																dev_result_file,
																FLAGS.with_char,
																FLAGS.char_len)
	elif FLAGS.tokenizer_type == "full_bpe":
		write_to_tfrecords.convert_bert_distillation_classifier_examples_to_features(train_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																train_result_file,
																FLAGS.with_char,
																FLAGS.char_len,
																label_type=FLAGS.label_type)

		write_to_tfrecords.convert_bert_distillation_classifier_examples_to_features(dev_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																dev_result_file,
																FLAGS.with_char,
																FLAGS.char_len,
																label_type=FLAGS.label_type)

		test_examples = classifier_data_api.get_train_examples(test_file,
											is_shuffle=False)
		write_to_tfrecords.convert_bert_distillation_classifier_examples_to_features(test_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer_corpus,
																test_result_file,
																FLAGS.with_char,
																FLAGS.char_len,
																label_type=FLAGS.label_type) 

	# elif FLAGS.if_rule == "rule":
	# 	print("==apply rule==")
	# 	with open(FLAGS.rule_word_path, "r") as frobj:
	# 		lines = frobj.read().splitlines()
	# 		freq_dict = []
	# 		for line in lines:
	# 			content = line.split("&&&&")
	# 			word = "".join(content[0].split("&"))
	# 			label = "rule"
	# 			tmp = {}
	# 			tmp["word"] = word
	# 			tmp["label"] = "rule"
	# 			freq_dict.append(tmp)
	# 		print(len(freq_dict))
	# 		json.dump(freq_dict, open(FLAGS.rule_word_dict, "w"))
	# 	from data_generator import rule_detector

	# 	# label_dict = {"label2id":{"正常":0,"rule":1}, "id2label":{0:"正常", 1:"rule"}}
	# 	# json.dump(label_dict, open("/data/xuht/websiteanalyze-data-seqing20180821/data/rule/rule_label_dict.json", "w"))

	# 	rule_config = {
	# 		"keyword_path":FLAGS.rule_word_dict,
	# 		"background_label":"正常",
	# 		"label_dict":FLAGS.rule_label_dict
	# 	}
	# 	rule_api = rule_detector.RuleDetector(rule_config)
	# 	rule_api.load(tokenizer)

	# 	classifier_data_api = classifier_processor.PornClassifierProcessor()
	# 	classifier_data_api.get_labels(FLAGS.label_id)

	# 	train_examples = classifier_data_api.get_train_examples(
	# 		FLAGS.train_file)

	# 	write_to_tfrecords.convert_classifier_examples_with_rule_to_features(train_examples,
	# 															classifier_data_api.label2id,
	# 															FLAGS.max_length,
	# 															tokenizer,
	# 															rule_api,
	# 															FLAGS.train_result_file)

	# 	test_examples = classifier_data_api.get_train_examples(FLAGS.test_file)
	# 	write_to_tfrecords.convert_classifier_examples_with_rule_to_features(test_examples,
	# 															classifier_data_api.label2id,
	# 															FLAGS.max_length,
	# 															tokenizer,
	# 															rule_api,
	# 															FLAGS.test_result_file)


if __name__ == "__main__":
	tf.app.run()

