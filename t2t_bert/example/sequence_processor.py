from data_generator import tf_data_utils
from data_generator import data_processor
from data_generator.tokenization import WordpieceTokenizer
from data_generator import tokenization

import csv
import json
import collections
import tensorflow as tf
import numpy as np

import random

import pandas as pd
import re
from hanziconv import HanziConv

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

def clean(text):
	text = text.strip()
	# text = text.lower()
	text = HanziConv.toSimplified(text)
	text = full2half(text)
	text = re.sub("\\#.*?#|\\|.*?\\||\\[.*?]", "", text)
	# text = re.sub("\s*", "", text)
	return text

CN_CHARACTER_REGEX = re.compile(u"[\u4e00-\u9fa5]+")

class ProductTitleLanguageModelProcessor(object):
	def _read_write(self, input_file, output_file, tokenizer,
					max_length=64,
					bos='<S>', eos='<T>', **kargs):
		self._writer = tf.python_io.TFRecordWriter(output_file)
		with tf.gfile.Open(input_file, "r") as f:
			for i, line in enumerate(f):
				if not line.strip():
					continue
				content = clean(line.strip())
				word_seq = []

				if kargs.get('token_mapping', {}):
					for key in kargs.get('token_mapping', {}):
						content = re.sub(key, kargs.get('token_mapping', {}).get(key, ""), content)

				for word in content.split():
					if CN_CHARACTER_REGEX.findall(word):
						word_seq.extend(list(word))
					else:
						word_seq.append(word)

				word_seq = [bos] + word_seq + [eos]
				word_id_seq = tokenizer.convert_tokens_to_ids(word_seq, max_length+2)
				seq_mask = [1] * len(word_id_seq)
				word_id_seq = tokenizer.padding(word_id_seq, max_length+2, 0)
				seq_mask = tokenizer.padding(seq_mask, max_length+2, 0)

				features = collections.OrderedDict()
				features["input_ids"] = tf_data_utils.create_int_feature(word_id_seq)
				features["input_mask"] = tf_data_utils.create_int_feature(seq_mask)

				if i <= 30:
					tf.logging.info("*** Example ***")
					tf.logging.info("input_ids: %s" % " ".join([str(x) for x in word_id_seq]))
					tf.logging.info("input_ids_ori: %s" % " ".join(word_seq))
			
				tf_example = tf.train.Example(features=tf.train.Features(feature=features))
				self._writer.write(tf_example.SerializeToString())

		self._writer.close()