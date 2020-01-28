# -*- coding: utf-8 -*-
from data_generator import data_processor
import collections
import re
import tensorflow as tf
from data_generator import tokenization
from hanziconv import HanziConv
import random
import platform
import unicodedata

def full2half(text):
	text = unicodedata.normalize('NFKC', text)
	return text

def clean(text):
	text = text.strip()
	text = tokenization.convert_to_unicode(text)
	text = HanziConv.toSimplified(text)
	text = full2half(text)
	text = re.sub(u"\\#.*?#|\\|.*?\\||\\[.*?]", "", text)
	text = re.sub(u"\\s*", "", text)
	return text

ExampleInstance = collections.namedtuple("ExampleInstance",
										  ["guid", "text_a",
										  "text_b",
										  "label"])

class SentenceProcessor(data_processor.DataProcessor): 
	def get_labels(self, label_file):
		import json
		with tf.gfile.Open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]
	
	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				content = line.strip()
				lines.append(content)
			return lines

	def _create_examples(self, lines,
								LABEL_SPLITTER=u"__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\\d+")
		label_pattern = u"(?<={})(\\d+)".format(LABEL_SPLITTER)
		
		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])

				input_labels = []
				for l in re.finditer(label_pattern, line):
					input_labels.append(l.group())

				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				examples.append(ExampleInstance(
                    	guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels))
			except:
				print(line, i)
		return examples

	def get_train_examples(self, train_file, is_shuffle=True):
		data = self._read_data(train_file)
		examples = self._create_examples(data)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle=False):
		data = self._read_data(dev_file)
		examples = self._create_examples(data)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_test_examples(self, test_file):
		data = self._read_data(test_file)
		examples = self._create_examples(data)
		return examples

class SentencePairProcessor(data_processor.DataProcessor): 
	def get_labels(self, label_file):
		import json
		with tf.gfile.Open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]
	
	def _read_data(self, input_file):
		import json
		data = []
		with tf.gfile.Open(input_file, "r") as frobj:
			for line in frobj:
				data.append(json.loads(line.strip()))
		return data

	def _create_examples(self, data):
		examples = []
		for index in range(len(data)):
			content = data[index]
			try:
				guid = int(content["ID"])
			except:
				guid = index
			try:
				text_a = clean(content["sentence1"])
				text_b = clean(content["sentence2"])
			except:
				print(content["sentence1"], content["sentence2"], index)
			try:
				label = content["gold_label"]
			except:
				label = "0"
			if isinstance(text_a, str) and isinstance(text_b, str) or isinstance(text_a, unicode) and isinstance(text_b, unicode):
				examples.append(ExampleInstance(
                    	guid=guid,
						text_a=text_a,
						text_b=text_b,
						label=[label]))
		return examples

	def get_train_examples(self, train_file, is_shuffle=True):
		data = self._read_data(train_file)
		examples = self._create_examples(data)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle=False):
		data = self._read_data(dev_file)
		examples = self._create_examples(data)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_test_examples(self, test_file):
		data = self._read_data(test_file)
		examples = self._create_examples(data)
		return examples