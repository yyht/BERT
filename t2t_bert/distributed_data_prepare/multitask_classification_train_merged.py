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

from example import feature_writer, write_to_tfrecords_multitask
from example import classifier_processor
from data_generator import vocab_filter
from collections import OrderedDict
import random

flags = tf.flags

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

## Required parameters
flags.DEFINE_string(
	"multitask_dict", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"vocab_file", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"lower_case", "",
	"if apply rule detector"
	)

flags.DEFINE_integer(
	"max_length", 128,
	"if apply rule detector"
	)

flags.DEFINE_string(
	"multi_task_type", "",
	"if apply rule detector"
	)

flags.DEFINE_string(
	"output_path", "",
	"if apply rule detector"
	)

def main(_):

	import json
	multi_task_config = Bunch(json.load(open(os.path.join(FLAGS.buckets, FLAGS.multitask_dict))))

	vocab_path = FLAGS.vocab_file
	# os.path.join(FLAGS.buckets, FLAGS.vocab_file)

	train_file_dict = {}
	test_file_dict = {}
	dev_file_dict = {}
	train_result_dict = {}
	test_result_dict = {}
	dev_result_dict = {}
	label_id_dict = {}
	for task in multi_task_config:
		train_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["train_file"])

		test_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["test_file"])

		dev_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["dev_file"])

		train_result_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["train_result_file"])

		test_result_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["test_result_file"])

		dev_result_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["dev_result_file"])
		label_id_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["label_id"])

	print(train_file_dict)

	if FLAGS.lower_case == "True":
		lower_case = True
	else:
		lower_case = False

	tokenizer = tokenization.FullTokenizer(
			vocab_file=vocab_path, 
			do_lower_case=lower_case)

	total_examples = []

	task_type_id = OrderedDict()
	label2id_dict = {}

	index = 0
	for task in (FLAGS.multi_task_type.split(",")):
		if task not in multi_task_config:
			continue
		task_type_id[task] = multi_task_config[task]
		index += 1
		data_type = multi_task_config[task]["data_type"]
		if data_type == "single_sentence":
			classifier_data_api = classifier_processor.SentenceProcessor()
			classifier_data_api.get_labels(label_id_dict[task])
		elif data_type == "sentence_pair":
			classifier_data_api = classifier_processor.SentencePairProcessor()
			classifier_data_api.get_labels(label_id_dict[task])

		train_examples = classifier_data_api.get_train_examples(train_file_dict[task],
											is_shuffle=True)
		label2id_dict[task] = classifier_data_api.label2id
		
		for item in train_examples:
			tmp = {"example":item,"task":task}
			total_examples.append(tmp)

	print(task_type_id.keys())
	print("==total data==", len(total_examples))

	random.shuffle(total_examples)
	write_to_tfrecords_multitask.convert_multitask_classifier_merged_examples_to_features(
			total_examples,
			label2id_dict,
			FLAGS.max_length,
			tokenizer,
			os.path.join(FLAGS.buckets, FLAGS.output_path),
			task_type_id
		)

if __name__ == "__main__":
	tf.app.run()

