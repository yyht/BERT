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
		"那王宝强还不如找个鸡呢，也花不到那么多的钱，马蓉的比是镶金边的吗",
		"SM粑粑威武霸气帅，JJ超级无敌粗又大",
		"叶可怡这时已全身赤裸一丝不挂",
		"仔细的舔弄丝袜脚上的高跟凉鞋和丝袜脚尖",
		"这车真耐操",
		"我是M，马奴，找美女S调教，付费。陪睡吗？",
		"享受她们美丽的肉体"
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



