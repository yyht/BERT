# -*- coding: utf-8 -*-
import sys,os,json

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

from distributed_single_sentence_classification import train_eval
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib

import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

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

flags.DEFINE_integer(
	"num_gpus", 2, 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"distribution_strategy", "MirroredStrategy", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"parse_type", "parse_single", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"rule_model", "normal", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"profiler", "normal", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"train_op", "adam", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"running_type", "eval", 
	"the required num_gpus"
	)

def main(_):

	print(FLAGS)
	print(tf.__version__, "==tensorflow version==")

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	dev_file = os.path.join(FLAGS.buckets, FLAGS.dev_file)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)

	print(init_checkpoint, train_file, dev_file, checkpoint_dir)

	sess_config = tf.ConfigProto(allow_soft_placement=True,
									log_device_placement=True)

	cluster = {'chief': ['localhost:2221'], 'worker': ['localhost:2222']}
	os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'evaluator', 'index': 0}})

	run_config = tf.estimator.RunConfig(
					  keep_checkpoint_max=5,
					  model_dir=checkpoint_dir, 
					  session_config=sess_config,
					  save_checkpoints_secs=None,
					  save_checkpoints_steps=None,
					  log_step_count_steps=100)

	task_index = run_config.task_id
	is_chief = run_config.is_chief
	worker_count = 1

	print("==worker_count==", worker_count, "==local_rank==", task_index, "==is is_chief==", is_chief)
	target = ""

	output_dict = []

	checkpoint_file = os.path.join(FLAGS.buckets, FLAGS.model_output, "checkpoint")
	
	with tf.gfile.GFile(checkpoint_file, 'r') as f:
		ckpts = f.readlines().split()
	for line in ckpts:
		if 'all_model_checkpoint_paths:' not in line:
			continue
		ckpt = '%s.index' % line.split(':')[-1].strip().strip('"')
		iteration = int(re.sub(r'^.*?(\d+).*', r'\1', ckpt))
		ckpt_path = checkpoint_dir.rstrip('\/')+'/'+ckpt
		if tf.gfile.Exists(ckpt_path):
			result_dict = train_eval.monitored_sess(FLAGS=FLAGS,
							worker_count=worker_count,
							task_index=task_index, 
							cluster=cluster, 
							is_chief=is_chief, 
							target=target,
							init_checkpoint=ckpt_path,
							train_file=train_file,
							dev_file=dev_file,
							checkpoint_dir=checkpoint_dir,
							run_config=run_config,
							profiler=FLAGS.profiler,
							parse_type=FLAGS.parse_type,
							rule_model=FLAGS.rule_model,
							train_op=FLAGS.train_op,
							running_type="eval")
			result_dict["index"] = iteration
			import json
			result_string = json.dumps(result_dict)
			output_dict.append(result_string)
	result_log_file = os.path.join(checkpoint_dir, "result.info")
	with tf.gfile.GFile(result_log_file, 'w') as f:
		for line in output_dict:
			fwobj.write(line+"\n")



