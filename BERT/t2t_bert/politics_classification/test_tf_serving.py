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

def main():

	query_lst = [
		"只要中国强大，全国人民团结，日本绝对不敢动中国，下一次，日本对我们会更狠。",
		"为人民币服务，交警没错",
		"支持曾江",
		"警察怎么查案的，都是白痴吗？",
		"我们这边情况更糟糕，低保都有指标，他愿意给谁就给谁，最后全给了他的亲人和哥们，以此巩固他的执政地位，谁告他，他就百般刁难你，玩死谁。",
		"敢是老百姓偷税漏税！你看早都收拾成啥了",
		"为什么人大代表中农民工代表比例越来越低了？而官商明星越来越多？"
	]

	features = []

	for query in query_lst:
		feature = get_single_features(query, 128)
		features.append(feature)

	if FLAGS.input_keys == "instances":
		for key in features[0]:
			import numpy as np
			print(np.array(features[0][key]).shape, key)
		feed_dict = {
			"instances":features[0:5],
			"signature_name":FLAGS.signature_name
		}

	elif FLAGS.input_keys == "inputs":
		feed_dict = {
			"inputs":{
				"input_ids":[],
				"input_mask":[],
				"segment_ids":[],
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



