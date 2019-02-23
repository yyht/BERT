import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
from data_generator import rule_detector
from model_io import model_io
import json

tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab, 
				do_lower_case=FLAGS.do_lower_case)

rule_config = {
    "keyword_path":FLAGS.keyword_path, #"/data/xuht/jd_comment/phrases.json",
    "background_label":FLAGS.background_label, #"正常",
    "label_dict":FLAGS.label_dict #"/data/xuht/jd_comment/rule_label_dict.json"
}
rule_api = rule_detector.RuleDetector(rule_config)
rule_api.load(tokenizer)

def full2half(ustring):
	rstring = ""
	for uchar in ustring:
		inside_code=ord(uchar)
		if inside_code==0x3000:
			inside_code=0x0020
		else:
			inside_code-=0xfee0
		if inside_code<0x0020 or inside_code>0x7e:   
			rstring += uchar
		else:
			rstring += unichr(inside_code)
	return rstring

def get_single_features(query, max_seq_length):
	tokens_a = tokenizer.tokenize(query)

	if len(tokens_a) > max_seq_length - 2:
		tokens_a = tokens_a[0:(max_seq_length - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)

	if FLAGS.segment_id_type == "normal":
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)
	elif FLAGS.segment_id_type == "rule":

		if os.path.exists(FLAGS.add_word_path):
			extra_word = json.load(open(FLAGS.add_word_path, "r"))
			for item in extra_word:
				rule_detector.keyword_detector.add_keyword(tokenizer.tokenize(item["word"]), [item["label"]])
		if os.path.exists(FLAGS.delete_word_path):
			delete_word = json.load(open(FLAGS.delete_word_path, "r"))
			for item in delete_word:
				rule_detector.keyword_detector.remove_keyword(tokenizer.tokenize(item["word"]))
		rule_ids = rule_detector.infer(tokens_a) # input is tokenized list
		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		for index, token in enumerate(tokens_a):
			tokens.append(token)
			segment_ids.append(rule_ids[index])
	else:
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
	label_ids = 0

	return {"input_ids":input_ids,
			"input_mask":input_mask,
			"segment_ids":segment_ids,
			"label_ids":[0]}

def get_feeddict(FLAGS):

	with open(FLAGS.query_path, "r") as frobj:
		query_dict_lst = json.load(frobj)

	query_lst = [item["query"] for item in query_dict_lst]
	label_lst = [item["label"] for item in query_dict_lst]

	features = []

	for query in query_lst:
		feature = get_single_features(query, FLAGS.max_length)
		features.append(feature)

	feed_dict = {
		"instances":features,
		"signature_name":FLAGS.signature_name,
	}

	return feed_dict



