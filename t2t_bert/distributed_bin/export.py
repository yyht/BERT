# -*- coding: utf-8 -*-
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

print(sys.path)

import tensorflow as tf

from distributed_single_sentence_classification import export as single_sentence_exporter

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_string("local_buckets", "", "oss buckets")

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_dir", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"export_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"export_type", None,
	"Input TF example files (can be a glob or comma separated).")

def export():

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	model_dir = os.path.join(FLAGS.buckets, FLAGS.model_dir)
	export_path = os.path.join(FLAGS.buckets, FLAGS.export_path)

	config_file = os.path.join(FLAGS.local_buckets, FLAGS.config_file)
	label_id = os.path.join(FLAGS.local_buckets, FLAGS.label_id)

	print(init_checkpoint, model_dir, export_path, "==load and store file on ==", FLAGS.buckets)
	print(config_file, label_id, "==load file from==", FLAGS.local_buckets)

	model_config = {
		"label2id":label_id,
		"init_checkpoint":init_checkpoint,
		"config_file":config_file,
		"max_length":FLAGS.max_length,
		"model_dir":model_dir,
		"export_path":export_path
	}
	if FLAGS.export_type == "1":
		single_sentence_exporter.export_model_v1(model_config)
	elif FLAGS.export_type == "2":
		single_sentence_exporter.export_model_v2(model_config)

