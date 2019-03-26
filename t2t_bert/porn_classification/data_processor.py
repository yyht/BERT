import sys,os,json
sys.path.append("..")

import numpy as np
import tensorflow as tf
from example import hvd_distributed_classifier as bert_classifier
from bunch import Bunch
from data_generator import tokenization
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
from model_io import model_io
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import horovod.tensorflow as hvd

from example import feature_writer, write_to_tfrecords
from porn_classification import classifier_processor

flags = tf.flags

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"test_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"train_result_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"test_result_file", None,
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

def main(_):

	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, 
		do_lower_case=FLAGS.lower_case)

	if FLAGS.if_rule != "rule":
		print("==not apply rule==")
		classifier_data_api = classifier_processor.PornClassifierProcessor()
		classifier_data_api.get_labels(FLAGS.label_id)

		train_examples = classifier_data_api.get_train_examples(FLAGS.train_file, is_shuffle=False)

		write_to_tfrecords.convert_classifier_examples_to_features(train_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer,
																FLAGS.train_result_file)

		test_examples = classifier_data_api.get_train_examples(FLAGS.test_file, is_shuffle=False)
		write_to_tfrecords.convert_classifier_examples_to_features(test_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer,
																FLAGS.test_result_file)
	elif FLAGS.if_rule == "rule":
		print("==apply rule==")
		with open(FLAGS.rule_word_path, "r") as frobj:
			lines = frobj.read().splitlines()
			freq_dict = []
			for line in lines:
				content = line.split("&&&&")
				word = "".join(content[0].split("&"))
				label = "rule"
				tmp = {}
				tmp["word"] = word
				tmp["label"] = "rule"
				freq_dict.append(tmp)
			print(len(freq_dict))
			json.dump(freq_dict, open(FLAGS.rule_word_dict, "w"))
		from data_generator import rule_detector

		label_dict = {"label2id":{"正常":0,"rule":1}, "id2label":{0:"正常", 1:"rule"}}
		json.dump(label_dict, open("/data/xuht/websiteanalyze-data-seqing20180821/data/rule/rule_label_dict.json", "w"))

		rule_config = {
			"keyword_path":FLAGS.rule_word_dict,
			"background_label":"正常",
			"label_dict":FLAGS.rule_label_dict
		}
		rule_api = rule_detector.RuleDetector(rule_config)
		rule_api.load(tokenizer)

		classifier_data_api = classifier_processor.PornClassifierProcessor()
		classifier_data_api.get_labels(FLAGS.label_id)

		train_examples = classifier_data_api.get_train_examples(
												FLAGS.train_file,
												 is_shuffle=True)

		write_to_tfrecords.convert_classifier_examples_with_rule_to_features(train_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer,
																rule_api,
																FLAGS.train_result_file)

		test_examples = classifier_data_api.get_train_examples(FLAGS.test_file,
													 is_shuffle=False)
		write_to_tfrecords.convert_classifier_examples_with_rule_to_features(test_examples,
																classifier_data_api.label2id,
																FLAGS.max_length,
																tokenizer,
																rule_api,
																FLAGS.test_result_file)


if __name__ == "__main__":
	tf.app.run()








