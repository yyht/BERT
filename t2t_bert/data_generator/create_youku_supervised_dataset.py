# -*- coding: utf-8 -*-

"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os,json
import sys,os

import time

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

import collections
import random
from heapq import nsmallest
import sys,os

from data_generator import tokenization
import tensorflow as tf

from collections import namedtuple
import multiprocessing
import numpy as np

from heapq import nsmallest
from itertools import accumulate
import random
import time, re
from data_generator import tf_data_utils
import jieba
from multiprocessing import Process, Manager

rng = random.Random(2008)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
		"buckets", None,
		"Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("input_file", None,
					"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
		"output_file", None,
		"Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
					"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("word_piece_model", None,
					"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
		"do_lower_case", True,
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 384, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
					"Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
		"dupe_factor", 10,
		"Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
		"short_seq_prob", 0.1,
		"Probability of creating sequences which are shorter than the "
		"maximum length.")

flags.DEFINE_bool(
		"do_whole_word_mask", False,
		"Probability of creating sequences which are shorter than the "
		"maximum length.")

flags.DEFINE_string("tokenizer_type", "bpe",
					"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("es_user_name", "mrc_search_4l",
					"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("password", "K9cb1bd713507",
					"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("doc_index", "mrc_pretrain",
					"Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("doc_type", "_doc",
					"Input raw text file (or comma-separated list of files).")

TrainingInstance = namedtuple("TrainingInstance",
										  ['tokens', 
										  'segment_ids',
										   'label_ids'])

def write_supervised_single_sintance_to_example_files(writer, instance, tokenizer, 
							max_seq_length, inst_index):
	input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
	# input_ori_ids = tokenizer.convert_tokens_to_ids(instance.original_tokens)
	input_mask = [1] * len(input_ids)
	segment_ids = list(instance.segment_ids)

	token_lens = len(input_ids)

	input_ids = tokenizer.padding(input_ids, max_seq_length, 0)
	input_mask = tokenizer.padding(input_mask, max_seq_length, 0)
	segment_ids = tokenizer.padding(segment_ids, max_seq_length, 0)

	# masked_lm_positions = list(instance.masked_lm_positions)
	# masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
	# masked_lm_weights = [1.0] * len(masked_lm_ids)

	# masked_lm_positions = tokenizer.padding(masked_lm_positions, max_predictions_per_seq,
	# 												0)
	# masked_lm_ids = tokenizer.padding(masked_lm_ids, max_predictions_per_seq,
	# 											0)
	# masked_lm_weights = tokenizer.padding(masked_lm_weights, max_predictions_per_seq,
	# 											0.0)

	features = collections.OrderedDict()
	features["input_ids"] = create_int_feature(input_ids)
	# features["input_ori_ids"] = create_int_feature(input_ori_ids)
	features["input_mask"] = create_int_feature(input_mask)
	features["segment_ids"] = create_int_feature(segment_ids)
	# features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
	# features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
	# features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
	features["label_ids"] = create_int_feature([instance.label_ids])

	tf_example = tf.train.Example(features=tf.train.Features(feature=features))

	writer.write(tf_example.SerializeToString())

	if inst_index < 10:
		tf.logging.info("*** Example ***")
		tf.logging.info("tokens: %s" % " ".join(
				[x for x in instance.tokens]))

		for feature_name in features.keys():
			feature = features[feature_name]
			values = []
			if feature.int64_list.value:
				values = feature.int64_list.value
			elif feature.float_list.value:
				values = feature.float_list.value
			tf.logging.info(
					"%s: %s" % (feature_name, " ".join([str(x) for x in values])))

def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

def get_training_isntance(document, max_seq_length):
	if not document:
		return []

	# index_range = list(range(num_of_documents))
	# index_range.remove(document_index)

	# random_document_lst = random.sample(index_range, len(index_range))
	
	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 3
	instances = []

	# We *usually* want to fill up the entire sequence since we are padding
	# to `max_seq_length` anyways, so short sequences are generally wasted
	# computation. However, we *sometimes*
	# (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
	# sequences to minimize the mismatch between pre-training and fine-tuning.
	# The `target_seq_length` is just a rough target however, whereas
	# `max_seq_length` is a hard limit.
	target_seq_length = max_num_tokens
	tokens_a_lst = []

	instances = []

	tokens_a = document['title']
	tokens_b = document['comment']
	label = document['label']

	tf_data_utils._truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		if token == '[UNK]':
			continue
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)
	for token in tokens_b:
		if token == '[UNK]':
			continue
		tokens.append(token)
		segment_ids.append(1)
	tokens.append("[SEP]")
	segment_ids.append(1)

	if label == "0":
		is_random_next = 0
	else:
		is_random_next = 1

	instance = TrainingInstance(
			tokens=tokens,
			segment_ids=segment_ids,
			label_ids=is_random_next)
	instances = [instance]
	return instances

def read_file(input_files, output_file, tokenizer, max_seq_length):
	"""Create `TrainingInstance`s from raw text."""
	# all_documents = [[]]

	# Input file format:
	# (1) One sentence per line. These should ideally be actual sentences, not
	# entire paragraphs or arbitrary spans of text. (Because we use the
	# sentence boundaries for the "next sentence prediction" task).
	# (2) Blank lines between documents. Document boundaries are needed so
	# that the "next sentence prediction" task doesn't span between documents.
	import json
	valid_doc_cnt = 0
	writer = tf.python_io.TFRecordWriter(output_file)
	for input_file in input_files:
		with tf.gfile.GFile(input_file, "r") as reader:
			for index, line in enumerate(reader):
				try:
					line = json.loads(line.strip())
				except:
					continue
				
				if not line:
					break
				title = line.get("title", None)
				content = line.get("comment", None)
				label = line.get("label", None)

				# Empty lines are used as document delimiters
				if not title or not content or not label:
					# all_documents.append([])
					continue
				# tf.logging.info(" line {}".format(line))

				title_cn_pattern = re.findall(CH_PATTERN, title)
				content_cn_pattern = re.findall(CH_PATTERN, content)
 
				if sum([len(item) for item in title_cn_pattern]) <= len(title) * 0.1:
					continue
				if sum([len(item) for item in content_cn_pattern]) <= len(content) * 0.1:
					continue

				valid_doc_cnt += 1

				tokens_title = tokenizer.tokenize(" ".join(jieba.cut(title)))
				tokens_content = tokenizer.tokenize(" ".join(jieba.cut(content)))
				
				document = {"title":tokens_title, 
							"comment":tokens_content,
							"label":label}

				instances = get_training_isntance(document, max_seq_length)
				for instance in instances:
					write_supervised_single_sintance_to_example_files(
							writer, instance, tokenizer, 
							max_seq_length, valid_doc_cnt)
	writer.close()
	tf.logging.info("Wrote %d total instances", valid_doc_cnt)

				
def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)

	print(FLAGS.do_whole_word_mask, FLAGS.do_lower_case)

	if FLAGS.tokenizer_type == "spm":
		word_piece_model = os.path.join(FLAGS.buckets, FLAGS.word_piece_model)
		tokenizer = tokenization.SPM(config={
			"word_dict":FLAGS.vocab_file,
			"word_piece_model":word_piece_model
			})
		tokenizer.load_dict()
		tokenizer.load_model()
		tokenizer.add_extra_word()
		tokenizer.build_word_id()
	elif FLAGS.tokenizer_type == "word_piece":
		tokenizer = tokenization.FullTokenizer(
			vocab_file=FLAGS.vocab_file, 
			do_lower_case=FLAGS.do_lower_case,
			do_whole_word_mask=FLAGS.do_whole_word_mask)

	input_files = []
	for input_pattern in FLAGS.input_file.split(","):
		input_files.extend(tf.gfile.Glob(input_pattern))

	tf.logging.info("*** Reading from input files ***")
	for input_file in input_files:
		tf.logging.info("  %s", input_file)

	rng = random.Random(FLAGS.random_seed)

	output_files = FLAGS.output_file.split(",")
	tf.logging.info("*** Writing to output files ***")
	for output_file in output_files:
		tf.logging.info("  %s", output_file)

	start = time.time()
	read_file(input_files, output_files[0], tokenizer, FLAGS.max_seq_length)
	
	print(time.time()-start, "==total time==")

if __name__ == "__main__":
	flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("word_piece_model")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("vocab_file")
	tf.app.run()


