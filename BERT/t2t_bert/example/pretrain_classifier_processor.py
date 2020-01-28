from data_generator import tf_data_utils
from data_generator import data_processor
from data_generator.tokenization import WordpieceTokenizer
from data_generator import tokenization
from data_generator import pretrain_feature
from data_generator import data_feature_mrc
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
	text = HanziConv.toSimplified(text)
	text = full2half(text)
	text = re.sub("\\#.*?#|\\|.*?\\||\\[.*?]", "", text)
	# text = re.sub("\s*", "", text)
	return text

class PiarPretrainProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]
	
	def _read_data(self, input_file):
		import json
		df = pd.read_csv(input_file)
		return df

	def _create_examples(self, df, lang="zh", debug=False):
		examples = []
		for index in range(df.shape[0]):
			content = df.loc[index]
			if content["tid1"] == content["tid2"]:
				continue
			guid = int(content["id"])
			if lang == "zh":
				text_a = content["title1_zh"]
				text_b = content["title2_zh"]
			elif lang == "en":
				text_a = content["title1_en"]
				text_b = content["title2_en"]
			label = content["label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append({"text_a":text_a,
								"text_b":text_b,
								"label":[label],
								"guid":guid})
			if debug and index == 100:
				break
		return examples

	def get_train_examples(self, train_file, lang="zh", debug=False):
		df = self._read_data(train_file)
		examples = self._create_examples(df, lang, debug)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, lang="zh", debug=False):
		df = self._read_data(dev_file)
		examples = self._create_examples(df, lang, debug)
		random.shuffle(examples)
		return examples

	def _create_test_examples(self, df, lang="zh"):
		examples = []
		for index in range(df.shape[0]):
			content = df.loc[index]
			guid = int(content["id"])
			if lang == "zh":
				text_a = content["title1_zh"]
				text_b = content["title2_zh"]
			elif lang == "en":
				text_a = content["title1_en"]
				text_b = content["title2_en"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append({"text_a":text_a,
								"text_b":text_b,
								"label":["unrelated"],
								"guid":guid})
		return examples

	def get_test_examples(self, test_file, lang="zh"):
		df = self._read_data(test_file)
		return self._create_test_examples(df, lang)

