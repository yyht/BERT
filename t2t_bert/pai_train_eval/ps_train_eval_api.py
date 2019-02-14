# -*- coding: utf-8 -*-
import sys,os
import os, sys

father_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
user_path = os.path.join(father_path, "pai_single_sentence_classification")

sys.path.append(father_path)
sys.path.append(user_path)


print(sys.path)

import tensorflow as tf

from pai_single_sentence_classification import ps_train_eval

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string('worker_hosts', '', 'must be list')
flags.DEFINE_string('job_name', '', 'must be in ("", "worker", "ps")')
flags.DEFINE_integer('task_index', 0, '')
flags.DEFINE_string("ps_hosts", "", "must be list")
flags.DEFINE_string("buckets", "", "oss buckets")

# flags.DEFINE_string(
# 	"config_file", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"init_checkpoint", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"vocab_file", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"label_id", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"max_length", 128,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"train_file", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"dev_file", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"model_output", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"epoch", 5,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"num_classes", 5,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"train_size", 1402171,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"batch_size", 32,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"model_type", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"if_shard", None,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_integer(
# 	"eval_size", 1000,
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"opt_type", "ps_sync",
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"is_debug", "0",
# 	"Input TF example files (can be a glob or comma separated).")

# flags.DEFINE_string(
# 	"run_type", "0",
# 	"Input TF example files (can be a glob or comma separated).")

def main(_):
	ps_spec = FLAGS.ps_hosts.split(",")
	worker_spec = FLAGS.worker_hosts.split(",")
	cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
	worker_count = len(worker_spec)

	is_chief = FLAGS.task_index == 0

	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
							protocol="grpc++")

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	dev_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)
	
	# join the ps server
	if FLAGS.job_name == "ps":
		server.join()
	try:
		if run_type == "sess":
			ps_train_eval.monitored_sess(
				worker_count=worker_count, 
				task_index=FLAGS.task_index, 
				cluster=cluster, 
				is_chief=is_chief, 
				target=server.target,
				init_checkpoint=init_checkpoint,
				train_file=train_file,
				dev_file=dev_file,
				checkpoint_dir=checkpoint_dir)

		elif run_type == "estimator":
			ps_train_eval.monitored_estimator(
				worker_count=worker_count, 
				task_index=FLAGS.task_index, 
				cluster=cluster, 
				is_chief=is_chief, 
				target=server.target,
				init_checkpoint=init_checkpoint,
				train_file=train_file,
				dev_file=dev_file,
				checkpoint_dir=checkpoint_dir)

	except Exception:
		print("catch a exception: %s" % "error")

if __name__ == "__main__":
	tf.app.run()