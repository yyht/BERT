import sys,os,json
sys.path.append("..")

import numpy as np
import tensorflow as tf
from bunch import Bunch
from example import feature_writer, write_to_tfrecords, classifier_processor
from data_generator import tokenization
from data_generator import tf_data_utils
from model_io import model_io
from example import write_to_records_pretrain

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

flags.DEFINE_integer(
	"num_threads", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_predictions_per_seq", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"log_cycle", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"feature_type", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_float(
	"masked_lm_prob", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_float(
	"dupe", None,
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, 
		do_lower_case=FLAGS.lower_case)

	classifier_data_api = classifier_processor.ClassificationProcessor()
	classifier_data_api.get_labels(FLAGS.label_id)

	train_examples = classifier_data_api.get_train_examples(FLAGS.train_file)

	write_to_records_pretrain.multi_process(
			examples=train_examples, 
			process_num=FLAGS.num_threads, 
			label_dict=classifier_data_api.label2id,
			tokenizer=tokenizer, 
			max_seq_length=FLAGS.max_length,
			masked_lm_prob=FLAGS.masked_lm_prob, 
			max_predictions_per_seq=FLAGS.max_predictions_per_seq, 
			output_file=FLAGS.train_result_file,
			dupe=FLAGS.dupe,
			random_seed=2018,
			feature_type=FLAGS.feature_type,
			log_cycle=FLAGS.log_cycle,
			per_seq_dupe_func=write_to_records_pretrain.per_seq_dupe_func
		)

	test_examples = classifier_data_api.get_train_examples(FLAGS.test_file)
	write_to_records_pretrain.multi_process(
			examples=test_examples, 
			process_num=FLAGS.num_threads, 
			label_dict=classifier_data_api.label2id,
			tokenizer=tokenizer, 
			max_seq_length=FLAGS.max_length,
			masked_lm_prob=FLAGS.masked_lm_prob, 
			max_predictions_per_seq=FLAGS.max_predictions_per_seq, 
			output_file=FLAGS.test_result_file,
			dupe=FLAGS.dupe,
			random_seed=2018,
			feature_type=FLAGS.feature_type,
			log_cycle=FLAGS.log_cycle,
			per_seq_dupe_func=write_to_records_pretrain.per_seq_dupe_func
		)

	print("==Succeeded in preparing masked lm with finetuning data for task-finetuning with masked lm regularization")

if __name__ == "__main__":
	tf.app.run()