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
from data_generator import flash_text

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

flags.DEFINE_string("keyword_path", "keyword_path",
					"Input raw text file (or comma-separated list of files).")

try:
	from data_generator import es_indexing
	config = {
		'username':FLAGS.es_user_name,
		'password':FLAGS.password,
		'es_url':'http://zsearch.alipay.com:9999'
	}
	es_api = es_indexing.ESSearch(config)
	try:
		es_api.delete(FLAGS.doc_index)
		es_api.create(FLAGS.doc_index)
		time.sleep(10)
		print("==delete old index and create new index==")
	except:
		es_api.create(FLAGS.doc_index)
		time.sleep(10)
		print("==create new index==")
except:
	es_api = None

import jieba
TrainingInstance = namedtuple("TrainingInstance",
										  ['original_tokens',
										  	'tokens', 'segment_ids',
										   'masked_lm_positions',
										   'masked_lm_labels',
										   'is_random_next'])

MaskedLmInstance = collections.namedtuple("masked_lm", ["index", "label"])

CH_PUNCTUATION = u"[＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
EN_PUNCTUATION = u"['!#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'.]"
CH_PATTERN = re.compile(u"[\u4e00-\u9fa5]+")
NUM_PATTERN = re.compile(u"[0-9]+")

def whole_word_mask(cand_indexes, token, i):
	if FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
		cand_indexes[-1].append(i)
	else:
		cand_indexes.append([i])
	return cand_indexes

def write_single_sintance_to_example_files(writer, instance, tokenizer, max_seq_length,
									max_predictions_per_seq, output_file, inst_index):
	input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
	input_ori_ids = tokenizer.convert_tokens_to_ids(instance.original_tokens)
	input_mask = [1] * len(input_ids)
	segment_ids = list(instance.segment_ids)

	token_lens = len(input_ids)

	input_ids = tokenizer.padding(input_ids, max_seq_length, 0)
	input_mask = tokenizer.padding(input_mask, max_seq_length, 0)
	segment_ids = tokenizer.padding(segment_ids, max_seq_length, 0)
	input_ori_ids = tokenizer.padding(input_ori_ids, max_seq_length, 0)

	masked_lm_positions = list(instance.masked_lm_positions)
	masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
	masked_lm_weights = [1.0] * len(masked_lm_ids)

	masked_lm_positions = tokenizer.padding(masked_lm_positions, max_predictions_per_seq,
													0)
	masked_lm_ids = tokenizer.padding(masked_lm_ids, max_predictions_per_seq,
												0)
	masked_lm_weights = tokenizer.padding(masked_lm_weights, max_predictions_per_seq,
												0.0)

	next_sentence_label = 1 if instance.is_random_next else 0
	features = collections.OrderedDict()
	features["input_ids"] = create_int_feature(input_ids)
	features["input_ori_ids"] = create_int_feature(input_ori_ids)
	features["input_mask"] = create_int_feature(input_mask)
	features["segment_ids"] = create_int_feature(segment_ids)
	features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
	features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
	features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
	features["next_sentence_labels"] = create_int_feature([next_sentence_label])

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

def first(iterable, condition = lambda x: True):
	"""
	Returns the first item in the `iterable` that
	satisfies the `condition`.

	If the condition is not given, returns the first item of
	the iterable.

	Raises `StopIteration` if no item satysfing the condition is found.

	>>> first( (1,2,3), condition=lambda x: x % 2 == 0)
	2
	>>> first(range(3, 100))
	3
	>>> first( () )
	Traceback (most recent call last):
	...
	StopIteration
	"""
	return next(x for x in iterable if condition(x))

def get_document(all_documents, es_api, doc_index):
	if es_api:
		search_body = {
			"query":{
				"match":{"doc_id":doc_index}
			}
		}
		result = es_api.search_doc(FLAGS.doc_index, search_body, threshold=0.1)
		if result:
			return json.loads(result[0]['source']['doc'])
		else:
			return []
	else:
		return all_documents[doc_index]

def create_instances_from_document(
		all_documents, document_index, vocab_words,
		max_seq_length, short_seq_prob,
		masked_lm_prob, max_predictions_per_seq,
		rng, num_of_documents):
	"""Creates `TrainingInstance`s for a single document."""
	document = get_document(all_documents, es_api, document_index)
	if not document:
		return []

	# index_range = list(range(num_of_documents))
	# index_range.remove(document_index)

	# random_document_lst = random.sample(index_range, len(index_range))
	
	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 2
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

	tokens_a = []
	for j in range(0, len(document)):
		if len(tokens_a) + len(document[j]) <= target_seq_length:
			tokens_a.extend(document[j])
		elif len(tokens_a) + len(document[j]) > target_seq_length:
			if len(tokens_a) >= 1:
				tokens_a_lst.append(tokens_a)
			tokens_a = []
			if len(document[j]) > target_seq_length:
				tokens_a.extend(document[j][0:target_seq_length-1])
			else:
				tokens_a.extend(document[j])

	if len(tokens_a) >= 1:
		tokens_a_lst.append(tokens_a)

	for tokens_a in tokens_a_lst:
		is_random_next = False

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

		(output_tokens, masked_lm_positions,
		 masked_lm_labels) = create_masked_lm_predictions(
				 tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
		instance = TrainingInstance(
				original_tokens=tokens,
				tokens=output_tokens,
				segment_ids=segment_ids,
				is_random_next=is_random_next,
				masked_lm_positions=masked_lm_positions,
				masked_lm_labels=masked_lm_labels)
		instances.append(instance)
	return instances

def valid_line(tokens):
	unk_cnt = sum([1 for item in tokens if item == "[UNK]"])
	if unk_cnt / (len(tokens)+1e-5) >= 0.05 or len(tokens) < 16:
		return False
	else:
		return True

def read_file(input_files, tokenizer, max_seq_length, keyword_path):
	"""Create `TrainingInstance`s from raw text."""
	# all_documents = [[]]

	# Input file format:
	# (1) One sentence per line. These should ideally be actual sentences, not
	# entire paragraphs or arbitrary spans of text. (Because we use the
	# sentence boundaries for the "next sentence prediction" task).
	# (2) Blank lines between documents. Document boundaries are needed so
	# that the "next sentence prediction" task doesn't span between documents.
	es_all_documents = [[]]

	valid_doc_cnt = 0
	for input_file in input_files:
		with tf.gfile.GFile(input_file, "r") as reader:
			while True:
				line = reader.readline()
				
				if not line:
					break
				line = line.strip()

				# Empty lines are used as document delimiters
				if not line:
					# all_documents.append([])
					es_all_documents.append([])
					continue
				# tf.logging.info(" line {}".format(line))

				tokens = tokenizer.tokenize(line)
				valid_flag = valid_line(tokens)

				if tokens and valid_flag:
					# all_documents[-1].append(tokens)
					es_all_documents[-1].append(tokens)
				if np.mod(len(es_all_documents), 1000) == 0:
					es_index_documents = []
					for item in es_all_documents:
						if not item:
							continue
						es_index_documents.append({
								"doc":json.dumps(item, ensure_ascii=False),
								"doc_id":valid_doc_cnt
							})
						valid_doc_cnt += 1
					es_api.index_batch_doc(FLAGS.doc_index, FLAGS.doc_type, es_index_documents, 1000)
					es_all_documents = [[]]
					doc_index_lst = []
					
	if len(es_all_documents) >= 1:
		es_index_documents = []
		for item in es_all_documents:
			if not item:
				continue
			es_index_documents.append({
					"doc":json.dumps(item, ensure_ascii=False),
					"doc_id":valid_doc_cnt
				})
			valid_doc_cnt += 1
		es_api.index_batch_doc(FLAGS.doc_index, FLAGS.doc_type, es_index_documents, 1000)
		es_all_documents = [[]]
	
	document_cnt = valid_doc_cnt

	return document_cnt

def create_instances_chunk_from_document(all_documents, document_index_chunk, 
		max_seq_length, masked_lm_prob, max_predictions_per_seq, 
		short_seq_prob, tokenizer, output_file, rng, num_of_documents, chunk_id):
	vocab_words = list(tokenizer.vocab.keys())
	writer = tf.python_io.TFRecordWriter(output_file)

	total_written = 0
	inst_index = 0

	print(len(document_index_chunk), "==document index_chunk==")

	for document_index in document_index_chunk:
		instances = create_instances_from_document(
							all_documents, document_index, vocab_words,
							max_seq_length, short_seq_prob,
							masked_lm_prob, max_predictions_per_seq,
							rng, num_of_documents)
		for instance in instances:
			write_single_sintance_to_example_files(writer, instance, 
									tokenizer, max_seq_length,
									max_predictions_per_seq, 
									output_file, inst_index)
			inst_index += 1
			total_written += 1

	writer.close()
	tf.logging.info("Wrote %d total instances %d", total_written, chunk_id)

def build_index_chunk(num_of_documents, process_num, dupe_factor):
	chunk_size = int(num_of_documents/process_num)
	print(chunk_size, "==chunk_size==")

	index_chunk = {}
	for dupe_index in range(dupe_factor):
		random_index = np.random.permutation(range(num_of_documents)).tolist()
		for i_index in range(process_num):
			start = i_index * chunk_size
			end = (i_index+1) * chunk_size
			if i_index in index_chunk:
				index_chunk[i_index].extend(random_index[start:end])
			else:
				index_chunk[i_index] = random_index[start:end]
	return index_chunk

def multi_process(input_files, tokenizer,
				max_seq_length,
				masked_lm_prob, 
				max_predictions_per_seq, 
				short_seq_prob,
				output_file,
				process_num,
				dupe_factor,
				random_seed=2018):

	chunk_num = process_num - 1

	num_of_documents = read_file(input_files, tokenizer, max_seq_length)
	# num_of_documents = len(all_documents)
	time.sleep(100)

	print(num_of_documents, dupe_factor)

	chunks = build_index_chunk(num_of_documents, process_num, dupe_factor)
	pool = multiprocessing.Pool(processes=process_num)

	all_documents_shared = []

	for chunk_id, chunk_key in enumerate(chunks):
		output_file_ = output_file + "/chunk_{}.tfrecords".format(chunk_id)
		print("#mask_language_model_multi_processing.length of chunk: {} ;file_name:{};chunk_id:{}".format(len(chunks[chunk_key]),output_file_,chunk_id))
		# create_instances_chunk_from_document(all_documents_shared, chunks[chunk_key], 
		# 			max_seq_length, masked_lm_prob, 
		# 			max_predictions_per_seq,
		# 			short_seq_prob, tokenizer, 
		# 			output_file_, rng, num_of_documents, chunk_id)
		# break
		pool.apply_async(create_instances_chunk_from_document,
			args=(all_documents_shared, chunks[chunk_key], 
					max_seq_length, masked_lm_prob, 
					max_predictions_per_seq,
					short_seq_prob, tokenizer, 
					output_file_, rng, num_of_documents, chunk_id)) # apply_async
	pool.close()
	pool.join()

def perform_span_level_mask(index_set, vocab_words, 
							masked_lms, covered_indexes, 
							output_tokens, tokens, rng):
	mask_flag, ori_flag, random_sample_flag = False, False, False

	for index in index_set:
		covered_indexes.add(index)

		masked_token = None
		# 80% of the time, replace with [MASK]
		if rng.random() < 0.8:
			masked_token = "[MASK]"
		else:
			# 10% of the time, keep original
			if rng.random() < 0.5:
				masked_token = tokens[index]
			# 10% of the time, replace with random word
			else:
				for i in range(10):
					masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
					cn_pattern = re.search(CH_PUNCTUATION, masked_token)
					en_pattern = re.search(EN_PUNCTUATION, masked_token)

					if cn_pattern or en_pattern:
						continue
					else:
						break

		output_tokens[index] = masked_token

		masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

	# if rng.random() < 0.8:
	# 	mask_flag = True
	# else:
	# 	if rng.random() < 0.5:
	# 		ori_flag = True
	# 	else:
	# 		random_sample_flag = True

	# for index in index_set:
	# 	covered_indexes.add(index)
	# 	if mask_flag:
	# 		masked_token = "[MASK]"
	# 	else:
	# 		if ori_flag:
	# 			masked_token = tokens[index]
	# 		else:
	# 			for i in range(10):
	# 				masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
	# 				cn_pattern = re.search(CH_PUNCTUATION, masked_token)
	# 				en_pattern = re.search(EN_PUNCTUATION, masked_token)
	# 				if cn_pattern or en_pattern:
	# 					continue
	# 				else:
	# 					break
	# 	output_tokens[index] = masked_token
	# 	masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

	return masked_lms, output_tokens, covered_indexes

def create_masked_lm_predictions(tokens, masked_lm_prob,
								max_predictions_per_seq, vocab_words, rng):
	"""Creates the predictions for the masked LM objective."""

	cand_indexes = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]" or token == '[UNK]':
			continue
		# if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
		# 	cand_indexes[-1].append(i)
		# else:
		# 	cand_indexes.append([i])
		cand_indexes = whole_word_mask(cand_indexes, token, i)

	rng.shuffle(cand_indexes)

	output_tokens = list(tokens)

	num_to_predict = min(max_predictions_per_seq,
											 max(1, int(round(len(tokens) * masked_lm_prob))))

	masked_lms = []
	covered_indexes = set()
	for index_set in cand_indexes:
		# if len(index_set) == 1:
		# 	continue
		if len(masked_lms) >= num_to_predict:
			break
		if len(masked_lms) + len(index_set) > num_to_predict:
			continue
		is_any_index_covered = False
		for index in index_set:
			if index in covered_indexes:
				is_any_index_covered = True
				break
		if is_any_index_covered:
			continue

		[masked_lms, 
		output_tokens, 
		covered_indexes] = perform_span_level_mask(index_set, 
												vocab_words, 
												masked_lms, 
												covered_indexes, 
												output_tokens,
												tokens, 
												rng)
	assert len(masked_lms) <= num_to_predict
	masked_lms = sorted(masked_lms, key=lambda x: x.index)

	masked_lm_positions = []
	masked_lm_labels = []
	for p in masked_lms:
		masked_lm_positions.append(p.index)
		masked_lm_labels.append(p.label)

	return (output_tokens, masked_lm_positions, masked_lm_labels)

# def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
# 	"""Truncates a pair of sequences to a maximum sequence length."""
# 	while True:
# 		total_length = len(tokens_a) + len(tokens_b)
# 		if total_length <= max_num_tokens:
# 			break

# 		trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
# 		assert len(trunc_tokens) >= 1

# 		# We want to sometimes truncate from the front and sometimes from the
# 		# back to add more randomness and avoid biases.
# 		if rng.random() < 0.5:
# 			del trunc_tokens[0]
# 		else:
# 			trunc_tokens.pop()

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

	multi_process(
			input_files=input_files, 
			tokenizer=tokenizer,
			max_seq_length=FLAGS.max_seq_length,
			masked_lm_prob=FLAGS.masked_lm_prob, 
			max_predictions_per_seq=FLAGS.max_predictions_per_seq, 
			short_seq_prob=FLAGS.short_seq_prob,
			output_file=output_file,
			process_num=1,
			dupe_factor=FLAGS.dupe_factor,
			random_seed=1234567
		)
	print(time.time()-start, "==total time==")

if __name__ == "__main__":
	flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("word_piece_model")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("vocab_file")
	tf.app.run()

