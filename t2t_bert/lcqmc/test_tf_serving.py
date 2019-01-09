import sys,os
sys.path.append("..")
import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
from model_io import model_io
import json
import requests

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "vocab", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "url", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "port", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "model_name", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


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

def get_single_features(query, sent, max_seq_length):
	query = full2half(query)
	tokens_a = tokenizer.tokenize(query)

	sent = full2half(sent)
	tokens_b = tokenizer.tokenize(sent)

	def get_input(input_tokens_a, input_tokens_b):
		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)

		for token in input_tokens_a:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		for token in input_tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		return [tokens, input_ids, 
				input_mask, segment_ids]

	[tokens_a_,
	input_ids_a, 
	input_mask_a, 
	segment_ids_a] = get_input(tokens_a, tokens_b)

	[tokens_b_,
	input_ids_b, 
	input_mask_b, 
	segment_ids_b] = get_input(tokens_b, tokens_a)

	return {"input_ids_a":input_ids_a,
			"input_mask_a":input_mask_a,
			"segment_ids_a":segment_ids_a,
			"input_ids_b":input_ids_b,
			"input_mask_b":input_mask_b,
			"segment_ids_b":segment_ids_b,
			"label_ids":[0]}

def main():
	tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab, 
				do_lower_case=True)


	query = u"银行转证券怎么转"
	candidate_lst = 10*[u"银行转证券怎么才能转过去"]

	features = []

	for candidate in candidate_lst:
		feature = get_single_features(query, candidate, 128)
		features.append(feature)

	feed_dict = {
		"inputs":{
			"input_ids_a":[],
			"input_mask_a":[],
			"segment_ids_a":[],
			"input_ids_b":[],
			"input_mask_b":[],
			"segment_ids_b":[],
			"label_ids":[]
		}
	}

	for feature in features:
		for key in feed_dict["inputs"]:
			feed_dict["inputs"][key].append(feature[key])

	url = "http://{}:{}/models/{}:predict".format(FLAGS.url, FLAGS.port, FLAGS.model_name)
	print("==serving url==", url)

	results = requests.post(url, json=feed_dict)
	print(results)

if __name__ == "__main__":
	main()



