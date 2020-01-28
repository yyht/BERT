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
import json

flags = tf.flags

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"word_piece_model", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"tokenizer_type", None,
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	input_config = {
		"word_piece_model":FLAGS.word_piece_model
	}
	if FLAGS.tokenizer_type == 'spm':
		tokenizer = tokenization.SPM(input_config)
		tokenizer.load_model()
	elif FLAGS.tokenizer_type == 'jieba':
		tokenizer = jieba
	fwobj = open(FLAGS.output_file, "w")
	with open(FLAGS.train_file, "r") as frobj:
		for line in frobj:
			content = line.strip()
			if len(content) >= 1:
				token_lst = my_spm.tokenize(content)
				fwobj.write(" ".join(token_lst)+"\n")
			else:
				fwobj.write("\n")
	fwobj.close()

if __name__ == "__main__":
	tf.app.run()