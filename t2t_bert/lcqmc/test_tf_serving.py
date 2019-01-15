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

flags.DEFINE_string(
	"signature_name", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"input_keys", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab, 
				do_lower_case=True)

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

	query = u"银行转证券怎么转"
	candidate_lst = [u"银行转证券怎么才能转过去", 
					u"证券怎么转银行",
					u"我无法从银行转账到证券怎么办",
					u"银行转证券失败",
					u"好好学习，天天向上"]

	features = []

	for candidate in candidate_lst:
		feature = get_single_features(query, candidate, 500)
		features.append(feature)

	if FLAGS.input_keys == "instances":
		for key in features[0]:
			import numpy as np
			print(np.array(features[0][key]).shape, key)
		feed_dict = {
			"instances":features[0:1],
			"signature_name":FLAGS.signature_name
		}
		import json
		json.dump(features, open("/data/xuht/LCQMC/test.json", "w"))
	elif FLAGS.input_keys == "inputs":
		feed_dict = {
			"inputs":{
				"input_ids_a":[],
				"input_mask_a":[],
				"segment_ids_a":[],
				"input_ids_b":[],
				"input_mask_b":[],
				"segment_ids_b":[],
				"label_ids":[]
			},
			"signature_name":FLAGS.signature_name
		}
		for feature in features[0:5]:
			for key in feed_dict["inputs"]:
				if key not in ["label_ids"]:
					feed_dict["inputs"][key].append(feature[key])
				else:
					feed_dict["inputs"][key].extend(feature[key])

		for key in feed_dict["inputs"]:
			print(key, np.array(feed_dict["inputs"][key]).shape)

	results = requests.post("http://%s:%s/v1/models/%s:predict" % (FLAGS.url, FLAGS.port, FLAGS.model_name), json=feed_dict)
	try:
		print(results.json())
	except:
		import json
		print(results.content)

if __name__ == "__main__":
	main()



