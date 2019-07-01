# -*- coding: utf-8 -*-
import sys,os,json
import sys,os

father_path = os.path.join(os.getcwd())
print(father_path, "==father path==")

def find_bert(father_path):
	if father_path.split("/")[-1] == "BERT":
		return father_path

	output_path = ""
	for fi in os.listdir(father_path):
		if fi == "BERT":
			output_path = os.path.join(father_path, fi)
			break
		else:
			if os.path.isdir(os.path.join(father_path, fi)):
				find_bert(os.path.join(father_path, fi))
			else:
				continue
	return output_path

bert_path = find_bert(father_path)
t2t_bert_path = os.path.join(bert_path, "t2t_bert")
sys.path.extend([bert_path, t2t_bert_path])

import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
import json, jieba, re

flags = tf.flags

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_folder", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_prefix", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"vocab_size", 50000,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_type", 'bpe',
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_float(
	"character_coverage", 0.9995,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"mining_sentence_size", 5000000,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"input_sentence_size", 5000000,
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	jieba_tokenization_file = FLAGS.train_file+".tmp"
	fwobj = open(jieba_tokenization_file, "w")
	char_pattern = re.compile(u"[\u4e00-\u9fa5]+")
	with open(FLAGS.train_file, "r") as frobj:
		for line in frobj:
			out = []
			result = list(jieba.cut(line.strip()))
			for word in result:
				word = list(word)
				char_cn = char_pattern.findall(word)
				if len(char_cn) >= 1:
					out.extend(word)
				else:
					out.append(word)
			fwobj.write(" ".join(out)+"\n")

	train_config = {
		"corpus":jieba_tokenization_file,
		"model_prefix":os.path.join(FLAGS.output_folder, FLAGS.model_prefix),
		"vocab_size":FLAGS.vocab_size,
		"model_type":FLAGS.model_type,
		"character_coverage":FLAGS.character_coverage,
		"mining_sentence_size":FLAGS.mining_sentence_size,
		"input_sentence_size":FLAGS.input_sentence_size
	}

	my_spm = tokenization.SPM({})
	my_spm.train_model(train_config)

if __name__ == "__main__":
	tf.app.run()