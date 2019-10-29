
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
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
from offline_debug.train_estimator import train_eval_fn

import tensorflow as tf
import json

flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_integer("task_index", 0, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", 'worker', "job name: worker or ps")

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
	"cross_tower_ops_type", "paisoar",
	"the CollectiveAllReduceStrategy cross_tower_ops_type"
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
	"train_op", "adam_weight_decay_exclude", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"running_type", "train", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"load_pretrained", "no", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"w2v_path", "",
	"pretrained w2v"
	)

flags.DEFINE_string(
	"with_char", "no_char",
	"pretrained w2v"
	)

flags.DEFINE_string(
	"input_target", "", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"decay", "no",
	"pretrained w2v"
	)

flags.DEFINE_string(
	"warmup", "no",
	"pretrained w2v"
	)

flags.DEFINE_string(
	"distillation", "normal",
	"if apply distillation"
	)

flags.DEFINE_float(
	"temperature", 2.0,
	"if apply distillation"
	)

flags.DEFINE_float(
	"distillation_ratio", 1.0,
	"if apply distillation"
	)

flags.DEFINE_integer(
	"num_hidden_layers", 12,
	"if apply distillation"
	)

flags.DEFINE_string(
	"task_type", "single_sentence_classification",
	"if apply distillation"
	)

flags.DEFINE_string(
	"classifier", "order_classifier",
	"if apply distillation"
	)

flags.DEFINE_string(
	"output_layer", "interaction",
	"if apply distillation"
	)

flags.DEFINE_integer(
	"char_limit", 5,
	"if apply distillation"
	)

flags.DEFINE_string(
	"mode", "single_task",
	"if apply distillation"
	)

flags.DEFINE_string(
	"multi_task_type", "wsdm",
	"if apply distillation"
	)

flags.DEFINE_string(
	"multi_task_config", "wsdm",
	"if apply distillation"
	)

flags.DEFINE_string(
	"task_invariant", "no",
	"if apply distillation"
	)

flags.DEFINE_float(
	"init_lr", 5e-5,
	"if apply distillation"
	)

flags.DEFINE_string(
	"multitask_balance_type", "data_balanced",
	"if apply distillation"
	)

flags.DEFINE_integer(
	"prefetch", 0,
	"if apply distillation"
	)

flags.DEFINE_string(
	"ln_type", 'postln',
	"if apply distillation"
	)

def make_distributed_info_without_evaluator():
	worker_hosts = FLAGS.worker_hosts.split(",")
	if len(worker_hosts) > 1:
		cluster = {"chief": [worker_hosts[0]],
			   "worker": worker_hosts[1:]}
	else:
		cluster = {"chief": [worker_hosts[0]]}

	if FLAGS.task_index == 0:
		task_type = "chief"
		task_index = 0
	else:
		task_type = "worker"
		task_index = FLAGS.task_index - 1
	return cluster, task_type, task_index

def dump_into_tf_config(cluster, task_type, task_index):
  os.environ['TF_CONFIG'] = json.dumps(
	  {'cluster': cluster,
	   'task': {'type': task_type, 'index': task_index}})

def main(_):

	print(FLAGS)
	print(tf.__version__, "==tensorflow version==")

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	dev_file = os.path.join(FLAGS.buckets, FLAGS.dev_file)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)

	print(init_checkpoint, train_file, dev_file, checkpoint_dir)

	if FLAGS.distribution_strategy == "MirroredStrategy":
		cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps("nccl", 10, 0, 0)
		distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus, 
												cross_tower_ops=cross_tower_ops)
		worker_count = FLAGS.num_gpus
	elif FLAGS.distribution_strategy == "CollectiveAllReduceStrategy":
		print("==apply collective all reduce strategy==")
		cluster, task_type, task_index = make_distributed_info_without_evaluator()
		print("==cluster==", cluster, "==task_type==", task_type, "==task_index==", task_index)
		dump_into_tf_config(cluster, task_type, task_index)
		distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
    							num_gpus_per_worker=1,
    							cross_tower_ops_type='horovod')
		worker_count = len(cluster['chief']) + len(cluster['worker'])
	else:
		cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps("nccl", 10, 0, 0)
		distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus, 
												cross_tower_ops=cross_tower_ops)
		worker_count = FLAGS.num_gpus

	sess_config = tf.ConfigProto(allow_soft_placement=True,
									log_device_placement=True)

	run_config = tf.estimator.RunConfig(
					  keep_checkpoint_max=10,
					  # model_dir=checkpoint_dir,
					  train_distribute=distribution, # tf 1.8
					  # distribute=distribution,     # tf 1.4
					  session_config=sess_config,
					  save_checkpoints_secs=None,
					  save_checkpoints_steps=None,
					  log_step_count_steps=100)
					  # disable_evaluation=True) # tf180

	task_index = run_config.task_id
	is_chief = run_config.is_chief

	print("==worker_count==", worker_count, "==local_rank==", task_index, "==is is_chief==", is_chief)
	cluster = ""
	target = ""
	
	train_eval_fn(
		FLAGS=FLAGS,
		worker_count=worker_count, 
		task_index=task_index, 
		cluster=cluster, 
		is_chief=is_chief,
		target=target,
		init_checkpoint=init_checkpoint,
		train_file=train_file,
		dev_file=dev_file,
		is_debug=FLAGS.is_debug,
		checkpoint_dir=checkpoint_dir,
		run_config=run_config,
		distribution_strategy=FLAGS.distribution_strategy,
		profiler=FLAGS.profiler,
		parse_type=FLAGS.parse_type,
		rule_model=FLAGS.rule_model,
		train_op=FLAGS.train_op,
		running_type=FLAGS.running_type,
		decay=FLAGS.decay,
		warmup=FLAGS.warmup,
		input_target=FLAGS.input_target,
		distillation=FLAGS.distillation,
		temperature=FLAGS.temperature,
		distillation_ratio=FLAGS.distillation_ratio)

if __name__ == "__main__":
	tf.app.run()