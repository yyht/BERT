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

def main(_):

	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, 
		do_lower_case=FLAGS.lower_case)

	classifier_data_api = classifier_processor.PornClassifierProcessor()
	classifier_data_api.get_labels(FLAGS.label_id)

	train_examples = classifier_data_api.get_train_examples(FLAGS.train_file)

	write_to_tfrecords.convert_classifier_examples_to_features(train_examples,
															classifier_data_api.label2id,
															FLAGS.max_length,
															tokenizer,
															FLAGS.train_result_file)

	test_examples = classifier_data_api.get_train_examples(FLAGS.test_file)
	write_to_tfrecords.convert_classifier_examples_to_features(test_examples,
															classifier_data_api.label2id,
															FLAGS.max_length,
															tokenizer,
															FLAGS.test_result_file)

if __name__ == "__main__":
	tf.app.run()








