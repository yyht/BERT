import numpy as np
import tensorflow as tf
from example import bert_classifier_estimator

from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils

from model_io import model_io
from example import feature_writer, write_to_tfrecords, classifier_processor
import json
from data_generator import tokenization

import os



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

class InferAPI(object):
	def __init__(self, config):
		self.config = config

	def load_label_dict(self):
		with open(self.config["label2id"], "r") as frobj:
			self.label_dict = json.load(frobj)

	def init_model(self):

		self.graph = tf.Graph()
		with self.graph.as_default():

			init_checkpoint = self.config["init_checkpoint"]
			bert_config = json.load(open(self.config["bert_config"], "r"))

			self.model_config = Bunch(bert_config)
			self.model_config.use_one_hot_embeddings = True
			self.model_config.scope = "bert"
			self.model_config.dropout_prob = 0.1
			self.model_config.label_type = "single_label"

			opt_config = Bunch({"init_lr":2e-5, "num_train_steps":1e30, "cycle":False})
			model_io_config = Bunch({"fix_lm":False})

			self.num_classes = len(self.label_dict["id2label"])
			self.max_seq_length = self.config["max_length"]

			self.tokenizer = tokenization.FullTokenizer(
				vocab_file=self.config["bert_vocab"], 
				do_lower_case=True)

			self.sess = tf.Session()
			self.model_io_fn = model_io.ModelIO(model_io_config)
	
			model_fn = bert_classifier_estimator.classifier_model_fn_builder(
											self.model_config, 
											self.num_classes, 
											init_checkpoint, 
											reuse=None, 
											load_pretrained=True,
											model_io_fn=self.model_io_fn,
											model_io_config=model_io_config, 
											opt_config=opt_config)

			estimator_config = tf.estimator.RunConfig()
			self.estimator = tf.estimator.Estimator(
        				model_fn=model_fn,
        				model_dir=self.config["model_dir"],
        				config=estimator_config)

	def get_input_features(self, sent_lst):
		input_ids_lst, input_mask_lst, segment_ids_lst = [], [], []
		label_ids_lst = []
		for sent in sent_lst:
			sent = full2half(sent)
			tokens_a = self.tokenizer.tokenize(sent)
			if len(tokens_a) > self.max_seq_length - 2:
				tokens_a = tokens_a[0:(self.max_seq_length - 2)]

			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)

			for token in tokens_a:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < self.max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
			label_ids = 0

			input_ids_lst.append(input_ids)
			input_mask_lst.append(input_mask)
			segment_ids_lst.append(segment_ids)
			label_ids_lst.append(label_ids)

		return  {"input_ids":np.array(input_ids_lst).astype(np.int32),
			"input_mask":np.array(input_mask_lst).astype(np.int32),
			"segment_ids":np.array(segment_ids_lst).astype(np.int32),
			"label_ids":np.array(label_ids_lst).astype(np.int32)}

	def input_fn(self, input_features):
		dataset = tf.data.Dataset.from_tensor_slices(input_features)
		dataset = dataset.batch(self.config.get("batch_size", 20))
		return dataset

	def infer(self, sent_lst):
		with self.graph.as_default():
			input_features = self.get_input_features(sent_lst)
			output = []
			for result in self.estimator.predict(input_fn=lambda:  self.input_fn(input_features)):
				output.append(result)


			for item in output:
				item["label"] = self.label_dict["id2label"][str(item["pred_label"])]
			
			return output

	# def infer(self, sent_lst):
	# 	with self.graph.as_default():
	# 		input_features = self.get_input_features(sent_lst)
			
	# 		features = self.input_fn(input_features)
			
	# 		output = self.estimator.predict(input_fn=features)
			
	# 		for result in output:
	# 			print(result)
	# 		return output

		# 	label_id = []
		# 	label = []
		# 	prob = []
		# 	while True:
		# 		try:
		# 			eval_result = self.sess.run(result)
		# 			label_id.extend(eval_result["label_ids"])
		# 			label.extend(eval_result["pred_label"])
		# 			prob.extend(eval_result["prob"].tolist())
		# 		except tf.errors.OutOfRangeError:
		# 			print("End of dataset")
		# 			break
		# assert len(sent_lst) == len(label)
		# assert len(prob) == len(label)

		# id2label_lst = [self.label_dict["id2label"][str(idx)] for idx in label]
		# return 	{"sent":sent_lst,
		# 		"prob":prob,
		# 		"label":id2label_lst}   






