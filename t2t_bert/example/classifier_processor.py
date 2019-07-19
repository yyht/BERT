from data_generator import tf_data_utils
from data_generator import data_processor
from data_generator.tokenization import WordpieceTokenizer
from data_generator import tokenization
from data_generator import data_feature_classifier
from data_generator import data_distillation_feature_classifier
from data_generator import data_structure_distillation
from data_generator import data_feature_mrc
from data_generator import data_adv_adaptation
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

class ClassificationProcessor(data_processor.DataProcessor):

	def get_labels(self, label_file):
		import json
		with tf.gfile.Open(label_file, "r") as f:
			label_mappings = json.load(f)
			self.label2id = label_mappings["label2id"]

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		examples = []
		for (i, line) in enumerate(lines):
			guid = i
			element_list = line.split(LABEL_SPLITTER)
			text_a = tokenization.convert_to_unicode(element_list[0].strip())
			text_a = clean(text_a)
			input_labels = element_list[1:]
			input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
			
			examples.append(data_feature_classifier.InputExample(
					guid=guid,
					text_a=text_a,
					text_b=None,
					label=input_labels
				))
		return examples

	def get_train_examples(self, train_file):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		random.shuffle(examples)
		return examples

class MultiChoiceProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		return [0, 1, 2] # answer choice

	def _read_data(self, input_file):
		import json
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(json.loads(line.strip()))
			return lines

	def _create_examples(self, lines):
		examples = []
		choice_cnt = {}
		max_length = 0
		for (i, line) in enumerate(lines):
			try:
				qas_id = int(line["query_id"])
				query = tokenization.convert_to_unicode(line["query"])
				query = clean(query)
				answer = tokenization.convert_to_unicode(line["answer"])
				answer = clean(answer)
				answer_choice = tokenization.convert_to_unicode(line["alternatives"]).split("|")
				answer_choice = [clean(ans) for ans in answer_choice]
				answer_choice = list(set(answer_choice))
				random.shuffle(answer_choice)
				assert len(answer_choice) == 3
				context = tokenization.convert_to_unicode(line["passage"])
				context = clean(context)
				for index, ans in enumerate(answer_choice):
					if ans == answer:
						choice = index
						break
				if choice in choice_cnt:
					choice_cnt[choice] += 1
				else:
					choice_cnt[choice] = 1
				examples.append(data_feature_mrc.InputExample(
						qas_id=qas_id,
						question_text=query,
						doc_tokens=context,
						answer_choice=answer_choice,
						choice=choice
					))
			except:
				continue
		print(choice_cnt)
		return examples

	def get_train_examples(self, train_file):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		random.shuffle(examples)
		return examples

	def _create_eval_examples(self, lines):
		examples = []
		max_length = 0
		for (i, line) in enumerate(lines):
			try:
				qas_id = int(line["query_id"])
				query = tokenization.convert_to_unicode(line["query"])
				query = clean(query)
				answer_choice = tokenization.convert_to_unicode(line["alternatives"]).split("|")
				answer_choice = [clean(ans) for ans in answer_choice]
				answer_choice = list(set(answer_choice))
				assert len(answer_choice) == 3
				# random.shuffle(answer_choice)
				context = tokenization.convert_to_unicode(line["passage"])
				context = clean(context)
				examples.append(data_feature_mrc.InputExample(
						qas_id=qas_id,
						question_text=query,
						doc_tokens=context,
						answer_choice=answer_choice,
						choice=0
					))
			except:
				continue
		return examples

	def get_eval_examples(self, dev_file):
		lines = self._read_data(dev_file)
		return self._create_eval_examples(lines)

class PiarChoiceProcessor(data_processor.DataProcessor): 
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

	def _create_examples(self, df, lang="zh"):
		examples = []
		for index in range(df.shape[0]):
			content = df.loc[index]
			guid = int(content["id"])
			if content["tid1"] == content["tid2"]:
				continue
			if lang == "zh":
				text_a = content["title1_zh"]
				text_b = content["title2_zh"]
			elif lang == "en":
				text_a = content["title1_en"]
				text_b = content["title2_en"]
			label = content["label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label]
				))
		return examples

	def get_train_examples(self, train_file, lang="zh"):
		df = self._read_data(train_file)
		examples = self._create_examples(df, lang)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, lang="zh"):
		df = self._read_data(dev_file)
		examples = self._create_examples(df, lang)
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
			if isinstance(text_a, str) and isinstance(text_b, str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=["unrelated"]
				))
		return examples

	def get_test_examples(self, test_file, lang="zh"):
		df = self._read_data(test_file)
		return self._create_test_examples(df, lang)

class PiarInteractionProcessor(data_processor.DataProcessor): 
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

	def _create_examples(self, df, lang="zh"):
		examples = []
		for index in range(df.shape[0]):
			content = df.loc[index]
			guid = int(content["id"])
			if content["tid1"] == content["tid2"]:
				continue
			if lang == "zh":
				text_a = content["title1_zh"]
				text_b = content["title2_zh"]
			elif lang == "en":
				text_a = content["title1_en"]
				text_b = content["title2_en"]
			label = content["label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label]
				))
		return examples

	def get_train_examples(self, train_file, lang="zh"):
		df = self._read_data(train_file)
		examples = self._create_examples(df, lang)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, lang="zh"):
		df = self._read_data(dev_file)
		examples = self._create_examples(df, lang)
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
			if isinstance(text_a, str) and isinstance(text_b, str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=["unrelated"]
				))
		return examples

	def get_test_examples(self, test_file, lang="zh"):
		df = self._read_data(test_file)
		return self._create_test_examples(df, lang)

class PiarOrderProcessor(data_processor.DataProcessor): 
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

	def _create_examples(self, df, lang="zh"):
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
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label]
				))
		return examples

	def get_train_examples(self, train_file, lang="zh"):
		df = self._read_data(train_file)
		examples = self._create_examples(df, lang)
		random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, lang="zh"):
		df = self._read_data(dev_file)
		examples = self._create_examples(df, lang)
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
			if isinstance(text_a, str) and isinstance(text_b, str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=["unrelated"]
				))
		return examples

	def get_test_examples(self, test_file, lang="zh"):
		df = self._read_data(test_file)
		return self._create_test_examples(df, lang)

class LCQMCProcessor(data_processor.DataProcessor): 
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]
	
	def _read_data(self, input_file):
		import json
		data = []
		with tf.gfile.Open(input_file, "r") as frobj:
			for line in frobj:
				data.append(json.loads(line.strip()))
		# with open(input_file, "r") as frobj:
		# 	for line in frobj:
		# 		data.append(json.loads(line))
		return data

	def _create_examples(self, data, lang="zh"):
		examples = []
		for index in range(len(data)):
			content = data[index]
			try:
				guid = int(content["ID"])
			except:
				guid = index
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label]
				))
		return examples

	def get_train_examples(self, train_file, lang="zh", is_shuffle=True):
		data = self._read_data(train_file)
		examples = self._create_examples(data, lang)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, lang="zh", is_shuffle=False):
		data = self._read_data(dev_file)
		examples = self._create_examples(data, lang)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def _create_test_examples(self, data, lang="zh"):
		examples = []
		for index in range(data.shape[0]):
			content = data[index]
			try:
				guid = int(content["id"])
			except:
				guid = index
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			if isinstance(text_a, str) and isinstance(text_b, str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=["0"]
				))
		return examples

	def get_test_examples(self, test_file, lang="zh"):
		data = self._read_data(test_file)
		return self._create_test_examples(data, lang)

class FasttextClassifierProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")
		label_pattern = "(?<={})(\d+)".format(LABEL_SPLITTER)

		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])

				input_labels = []
				for l in re.finditer(label_pattern, line):
					input_labels.append(l.group())

				text_a = tokenization.convert_to_unicode(text_a)
				# input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels
					))
				
			except:
				print(line, i)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

class FasttextDistillationProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def _read_distillation(self, input_file):
		import json
		with tf.gfile.Open(input_file, "r") as f:
			return json.load(f)["prob"]

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")

		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])
				input_labels = clean(element_list[1]).split(LABEL_SPLITTER)[-1]

				text_a = tokenization.convert_to_unicode(text_a)
				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=[1.0/len(self.label2id)]*len(self.label2id),
						label_ratio=1.0,
						distillation_ratio=0.0
					))
			except:
				print(line, i)
		return examples

	def _create_unsupervised_distillation_examples(self, lines, distillation_prob , LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")

		examples = []
		cnt = 0
		# assert len(lines) == len(distillation_prob)
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])
				input_labels = clean(element_list[1]).split(LABEL_SPLITTER)[-1]
			except:
				print(line, i)
				continue

			text_a = tokenization.convert_to_unicode(text_a)
			input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
			
			examples.append(data_distillation_feature_classifier.InputExample(
					guid=guid,
					text_a=text_a,
					text_b=None,
					label=input_labels,
					label_probs=distillation_prob[cnt],
					label_ratio=0.0,
					distillation_ratio=1.0
				))
			cnt += 1
		assert cnt == len(distillation_prob)

		return examples

	def _create_supervised_distillation_examples(self, lines, distillation_prob , LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")

		examples = []

		# assert len(lines) == len(distillation_prob)
		cnt = 0
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])
				input_labels = clean(element_list[1]).split(LABEL_SPLITTER)[-1]
			except:
				print(line, i)
				continue

			text_a = tokenization.convert_to_unicode(text_a)
			input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
			
			examples.append(data_distillation_feature_classifier.InputExample(
					guid=guid,
					text_a=text_a,
					text_b=None,
					label=input_labels,
					label_probs=distillation_prob[cnt],
					label_ratio=1.0,
					distillation_ratio=1.0
				))
			cnt += 1
		assert cnt == len(distillation_prob)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_unsupervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_unsupervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_supervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_supervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

class LCQMCDistillationProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
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

	def _read_distillation(self, input_file):
		import json
		with tf.gfile.Open(input_file, "r") as f:
			return json.load(f)["prob"]

	def _create_examples(self, data, lang="zh"):
		examples = []
		for index in range(len(data)):
			content = data[index]
			guid = int(content["ID"])
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label],
						label_probs=[1.0/len(self.label2id)]*len(self.label2id),
						label_ratio=1.0,
						distillation_ratio=0.0
					))
		return examples

	def _create_supervised_distillation_examples(self, lines, distillation_prob):

		examples = []
		cnt = 0
		for (i, line) in enumerate(lines):
			content = line
			guid = int(content["ID"])
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):

				text_a = tokenization.convert_to_unicode(text_a)
				text_b = tokenization.convert_to_unicode(text_b)
				input_labels = [label]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=text_b,
						label=input_labels,
						label_probs=distillation_prob[cnt],
						label_ratio=1.0,
						distillation_ratio=1.0
					))
				cnt += 1
		assert cnt == len(distillation_prob)
		return examples

	def _create_unsupervised_distillation_examples(self, lines, distillation_prob):

		examples = []
		cnt = 0
			
		for (i, line) in enumerate(lines):
			content = line
			guid = int(content["ID"])
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):

				text_a = tokenization.convert_to_unicode(text_a)
				text_b = tokenization.convert_to_unicode(text_b)
				input_labels = [label]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=text_b,
						label=input_labels,
						label_probs=distillation_prob[cnt],
						label_ratio=0.0,
						distillation_ratio=1.0
					))
			cnt += 1
		assert cnt == len(distillation_prob)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_unsupervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_unsupervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_supervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_supervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

class FasttextProductClassifierProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")

		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean("".join(element_list[-1].split()[1:]))
				input_labels = clean(element_list[1]).split(LABEL_SPLITTER)[-1]

				text_a = tokenization.convert_to_unicode(text_a)
				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels
					))
			except:
				print(line, i)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

class FasttextStructureDistillationProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def _read_distillation(self, input_file):
		from example.read_distillation_tfrecord import read_distilaltion
		distilaltion_dict_lst = read_distilaltion(input_file)
		print("==total size of distillation==", len(distilaltion_dict_lst))
		return distilaltion_dict_lst

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")
		label_pattern = "(?<={})(\d+)".format(LABEL_SPLITTER)

		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i
				
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])

				input_labels = []
				for l in re.finditer(label_pattern, line):
					input_labels.append(l.group())

				text_a = tokenization.convert_to_unicode(text_a)
				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]

				examples.append(data_structure_distillation.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=[1.0/len(self.label2id)]*len(self.label2id),
						label_ratio=1.0,
						distillation_ratio=0.0,
						feature=None
					))
			except:
				print(line, i)
		return examples

	def _create_unsupervised_distillation_examples(self, lines, distillation_dict_lst , LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")
		label_pattern = "(?<={})(\d+)".format(LABEL_SPLITTER)

		examples = []
		cnt = 0
		# assert len(lines) == len(distillation_prob)
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])

				input_labels = []
				for l in re.finditer(label_pattern, line):
					input_labels.append(l.group())

				text_a = tokenization.convert_to_unicode(text_a)
				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]["prob"],
						label_ratio=0.0,
						distillation_ratio=1.0,
						feature=distillation_dict_lst[cnt]["feature"]
					))
				cnt += 1
			except:
				print(line, i)
				continue
		assert cnt == len(distillation_dict_lst)

		return examples

	def _create_supervised_distillation_examples(self, lines, distillation_dict_lst, LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\d+")
		label_pattern = "(?<={})(\d+)".format(LABEL_SPLITTER)

		examples = []
		cnt = 0
		for (i, line) in enumerate(lines):
			try:
				guid = i
				element_list = re.split(re_pattern, line)
				text_a = clean(element_list[-1])

				input_labels = []
				for l in re.finditer(label_pattern, line):
					input_labels.append(l.group())
				text_a = tokenization.convert_to_unicode(text_a)
				input_labels = [label.strip() for label in input_labels if label.strip() in list(self.label2id.keys())]
				
				if len(input_labels) == 1:
					assert int(input_labels[0]) == distillation_dict_lst[cnt]["label_id"]

				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]["prob"],
						label_ratio=1.0,
						distillation_ratio=1.0,
						feature=distillation_dict_lst[cnt]["feature"]
					))
				cnt += 1
			except:
				print(line, i)
				continue
			
		assert cnt == len(distillation_dict_lst)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_unsupervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_unsupervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_supervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_supervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

class LCQMCStructureDistillationProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def _read_distillation(self, input_file):
		from example.read_distillation_tfrecord import read_distilaltion
		distilaltion_dict_lst = read_distilaltion(input_file)
		print("==total size of distillation==", len(distilaltion_dict_lst))
		return distilaltion_dict_lst

	def _read_data(self, input_file):
		import json
		data = []
		with tf.gfile.Open(input_file, "r") as frobj:
			for line in frobj:
				data.append(json.loads(line.strip()))
		return data

	def _create_examples(self, data, lang="zh"):
		examples = []
		for index in range(len(data)):
			content = data[index]
			try:
				guid = int(content["ID"])
			except:
				guid = index
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label],
						label_probs=[1.0/len(self.label2id)]*len(self.label2id),
						label_ratio=1.0,
						distillation_ratio=0.0
					))
		return examples

	def _create_supervised_distillation_examples(self, lines, distillation_dict_lst):

		examples = []
		cnt = 0
		for (i, line) in enumerate(lines):
			content = line
			try:
				guid = int(content["ID"])
			except:
				guid = i
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):

				text_a = tokenization.convert_to_unicode(text_a)
				text_b = tokenization.convert_to_unicode(text_b)
				input_labels = [label]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=text_b,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]['prob'],
						label_ratio=1.0,
						distillation_ratio=1.0
					))
				cnt += 1
		assert cnt == len(distillation_prob)
		return examples

	def _create_unsupervised_distillation_examples(self, lines, distillation_dict_lst):

		examples = []
		cnt = 0
			
		for (i, line) in enumerate(lines):
			content = line
			try:
				guid = int(content["ID"])
			except:
				guid = i
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			label = content["gold_label"]
			if isinstance(text_a,str) and isinstance(text_b,str):

				text_a = tokenization.convert_to_unicode(text_a)
				text_b = tokenization.convert_to_unicode(text_b)
				input_labels = [label]
				
				examples.append(data_distillation_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=text_b,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]['prob'],
						label_ratio=0.0,
						distillation_ratio=1.0
					))
			cnt += 1
		assert cnt == len(distillation_prob)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_unsupervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_unsupervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_supervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_supervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
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
			text_a = content["sentence1"]
			text_b = content["sentence2"]
			try:
				label = content["gold_label"]
			except:
				label = "0"
			if isinstance(text_a,str) and isinstance(text_b,str):
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=clean(text_a),
						text_b=clean(text_b),
						label=[label]
				))
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
				lines.append(line.strip())
			return lines

	def _create_examples(self, lines,
								LABEL_SPLITTER="__label__"):
		re_pattern = u"({}{})".format(LABEL_SPLITTER, "\\d+")
		label_pattern = "(?<={})(\\d+)".format(LABEL_SPLITTER)
		
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

				text_a = tokenization.convert_to_unicode(text_a)				
				examples.append(data_feature_classifier.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels
					))
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

class AdvAdaptationDistillationSentenceProcessor(data_processor.DataProcessor):
	def get_labels(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.label2id = label["label2id"]
		self.id2label = label["id2label"]

	def get_adv_mapping(self, label_file):
		import json
		with open(label_file, "r") as frobj:
			label = json.load(frobj)
		self.adv2id = label["adv2id"]
		self.id2adv = label["id2adv"]

	def _read_distillation(self, input_file):
		from example.read_distillation_tfrecord import read_distilaltion
		distilaltion_dict_lst = read_distilaltion(input_file)
		print("==total size of distillation==", len(distilaltion_dict_lst))
		return distilaltion_dict_lst

	def _read_data(self, input_file):
		with tf.gfile.Open(input_file, "r") as f:
			lines = []
			for line in f:
				lines.append(json.loads(line.strip()))
			return lines

	def _create_examples(self, lines,
									LABEL_SPLITTER="__label__"):
		examples = []
		for (i, line) in enumerate(lines):
			try:
				guid = i

				input_labels = [line['label']]
				text_a = clean(line['content'].strip())
				lang_label = line['lang']

				text_a = tokenization.convert_to_unicode(text_a)

				examples.append(data_adv_adaptation.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=[1.0/len(self.label2id)]*len(self.label2id),
						label_ratio=1.0,
						distillation_ratio=0.0,
						feature=None,
						adv_label=self.adv2id[lang_label]
					))
			except:
				print(line, i)
		return examples

	def _create_unsupervised_distillation_examples(self, lines, distillation_dict_lst , LABEL_SPLITTER="__label__"):

		examples = []
		cnt = 0
		# assert len(lines) == len(distillation_prob)
		for (i, line) in enumerate(lines):
			try:
				guid = i
				input_labels = [line['label']]
				text_a = clean(line['content'].strip())
				lang_label = line['lang']

				text_a = tokenization.convert_to_unicode(text_a)				
				examples.append(data_adv_adaptation.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]["prob"],
						label_ratio=0.0,
						distillation_ratio=1.0,
						feature=distillation_dict_lst[cnt]["feature"],
						adv_label=self.adv2id[lang_label]
					))
				cnt += 1
			except:
				print(line, i)
				continue
		assert cnt == len(distillation_dict_lst)

		return examples

	def _create_supervised_distillation_examples(self, lines, distillation_dict_lst, LABEL_SPLITTER="__label__"):
		examples = []
		cnt = 0
		for (i, line) in enumerate(lines):
			try:
				guid = i
				input_labels = [line['label']]
				text_a = clean(line['content'].strip())
				lang_label = line['lang']

				text_a = tokenization.convert_to_unicode(text_a)
				
				examples.append(data_adv_adaptation.InputExample(
						guid=guid,
						text_a=text_a,
						text_b=None,
						label=input_labels,
						label_probs=distillation_dict_lst[cnt]["prob"],
						label_ratio=1.0,
						distillation_ratio=1.0,
						feature=distillation_dict_lst[cnt]["feature"],
						adv_label=self.adv2id[lang_label]
					))
				cnt += 1
			except:
				print(line, i)
				continue
			
		assert cnt == len(distillation_dict_lst)
		return examples

	def get_train_examples(self, train_file, is_shuffle):
		lines = self._read_data(train_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_dev_examples(self, dev_file, is_shuffle):
		lines = self._read_data(dev_file)
		examples = self._create_examples(lines)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_unsupervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_unsupervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples

	def get_supervised_distillation_examples(self, dev_file, distillation_file, is_shuffle):
		distillation = self._read_distillation(distillation_file)
		lines = self._read_data(dev_file)
		examples = self._create_supervised_distillation_examples(lines, distillation)
		if is_shuffle:
			random.shuffle(examples)
		return examples




 	






