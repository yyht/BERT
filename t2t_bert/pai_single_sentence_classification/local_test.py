# -*- coding: utf-8 -*-
import sys,os

sys.path.append("..")

# father_path = os.path.join(os.getcwd())
# print(father_path, "==father path==")

# def find_bert(father_path):
# 	if father_path.split("/")[-1] == "BERT":
# 		return father_path

# 	output_path = ""
# 	for fi in os.listdir(father_path):
# 		if fi == "BERT":
# 			output_path = os.path.join(father_path, fi)
# 			break
# 		else:
# 			if os.path.isdir(os.path.join(father_path, fi)):
# 				find_bert(os.path.join(father_path, fi))
# 			else:
# 				continue
# 	return output_path

# bert_path = find_bert(father_path)
# t2t_bert_path = os.path.join(bert_path, "t2t_bert")
# sys.path.extend([bert_path, t2t_bert_path])

import tensorflow as tf
from ps_train_eval import monitored_estimator, monitored_sess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", 128,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"dev_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_output", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"epoch", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"num_classes", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"train_size", 1402171,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"batch_size", 32,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_type", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"if_shard", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"eval_size", 1000,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"opt_type", "ps_sync",
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"is_debug", "0",
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"run_type", "0",
	"Input TF example files (can be a glob or comma separated).")

def run(FLAGS):
	if FLAGS.run_type == "sess":
		monitored_sess(worker_count=1,
						task_index=0,
						cluster="",
						is_chief=True,
						target="",
						init_checkpoint=FLAGS.init_checkpoint,
						train_file=FLAGS.train_file,
						dev_file=FLAGS.dev_file,
						checkpoint_dir=FLAGS.model_output)
	elif FLAGS.run_type == "estimator":
		monitored_estimator(worker_count=1,
						task_index=0,
						cluster="",
						is_chief=True,
						target="",
						init_checkpoint=FLAGS.init_checkpoint,
						train_file=FLAGS.train_file,
						dev_file=FLAGS.dev_file,
						checkpoint_dir=FLAGS.model_output)

run(FLAGS)

