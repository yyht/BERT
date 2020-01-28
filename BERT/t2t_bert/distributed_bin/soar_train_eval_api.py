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

import sklearn
print("==sklearn version==", sklearn.__version__)

import tensorflow as tf
from distributed_single_sentence_classification import soar_train_eval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import paisoar

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('worker_hosts', '', 'must be list')
flags.DEFINE_string('job_name', '', 'must be in ("", "worker", "ps")')
flags.DEFINE_integer('task_index', 0, '')
flags.DEFINE_string("ps_hosts", "", "must be list")
flags.DEFINE_string("buckets", "", "oss buckets")

flags.DEFINE_string('taskId', None, '')
flags.DEFINE_integer('worker_count', None, '')
flags.DEFINE_integer('ps_count', None, '')

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

flags.DEFINE_bool(
  'log_device_placement', False,
  'Whether or not to log device placement.')

flags.DEFINE_bool(
  'use_fp16', True,
  'Use 16-bit floats for certain tensors instead of 32-bit floats')

flags.DEFINE_string('protocol', 'grpc', 
	'default grpc. if rdma cluster, use grpc+verbs instead.')

flags.DEFINE_integer(
  'inter_op_parallelism_threads', 256, 'Compute pool size')
flags.DEFINE_integer(
  'intra_op_parallelism_threads', 96, 'Eigen pool size')

flags.DEFINE_string("tensor_fusion_policy", 'default', '')
flags.DEFINE_string("communication_policy", 'nccl_fullring', '')
flags.DEFINE_integer("tensor_fusion_max_bytes", 32<<20, '')

flags.DEFINE_bool('enable_bfloat16_sendrecv', False, '')

try:
	import paisoar as pai
except Exception as e:
	pai = None

def get_cluster_manager(config_proto):
	"""Returns the cluster manager to be used."""
	return GrpcClusterManager(config_proto)

class BaseClusterManager(object):
	"""The manager for the cluster of servers running the fast-nn."""
	def __init__(self):
		assert FLAGS.job_name in ['worker', None, ''], 'job_name must be worker'
		if FLAGS.job_name and FLAGS.worker_hosts:
			cluster_dict = {'worker': FLAGS.worker_hosts.split(',')}
		else:
			cluster_dict = {'worker': ['127.0.0.1:0']}

		self._num_workers = len(cluster_dict['worker'])
		self._cluster_spec = tf.train.ClusterSpec(cluster_dict)
		self._device_exp = tf.train.replica_device_setter(
		  worker_device="/job:worker/task:%d/" % FLAGS.task_index,
		  cluster=self._cluster_spec)

	def get_target(self):
		"""Returns a target to be passed to tf.Session()."""
		raise NotImplementedError('get_target must be implemented by subclass')

	def get_cluster_spec(self):
		return self._cluster_spec

	def num_workers(self):
		return self._num_workers

	def device_exp(self):
		return self._device_exp

class GrpcClusterManager(BaseClusterManager):
	"""A cluster manager for a cluster networked with gRPC."""
	def __init__(self, config_proto):
		super(GrpcClusterManager, self).__init__()
		self._server = tf.train.Server(self._cluster_spec,
									   job_name=FLAGS.job_name,
									   task_index=FLAGS.task_index,
									   config=config_proto,
									   protocol=FLAGS.protocol)
		self._target = self._server.target

	def get_target(self):
		return self._target

def create_config_proto():
	"""Returns session config proto."""
	config = tf.ConfigProto(
	log_device_placement=FLAGS.log_device_placement,
	inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
	intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
	allow_soft_placement=True,
	gpu_options=tf.GPUOptions(
	  force_gpu_compatible=True,
	  allow_growth=True))
	config.graph_options.enable_bfloat16_sendrecv = FLAGS.enable_bfloat16_sendrecv
	return config

def main(_):

	print(tf.__version__, "==tensorflow version==")
	print(pai, "===pai")
	try:
		print(pai.__version__, "==pai soar version==")
	except:
		print("==no pai soar==")
	print("job name = %s" % FLAGS.job_name)
	print("task index = %d" % FLAGS.task_index)
	import os
	buckets_path = os.path.join("/buckets", "alg-misc", "BERT")
	os.system("ls {}".format(buckets_path))

	init_checkpoint = os.path.join(buckets_path, FLAGS.init_checkpoint)
	train_file = os.path.join(buckets_path, FLAGS.train_file)
	dev_file = os.path.join(buckets_path, FLAGS.dev_file)
	checkpoint_dir = os.path.join(buckets_path, FLAGS.model_output)

	print(init_checkpoint, train_file, dev_file, checkpoint_dir)

	is_chief = FLAGS.task_index == 0
	
	paisoar.enable_replicated_vars(tensor_fusion_policy=FLAGS.tensor_fusion_policy,
								   communication_policy=FLAGS.communication_policy,
								   tensor_fusion_max_bytes=FLAGS.tensor_fusion_max_bytes)

	sess_config = create_config_proto()

	cluster_manager = get_cluster_manager(config_proto=sess_config)

	target = cluster_manager.get_target()
	worker_count = cluster_manager.num_workers()
	cluster = ""

	with tf.device(cluster_manager.device_exp()):
		if FLAGS.run_type == "sess":
			print("==sess worker running==", FLAGS.job_name, FLAGS.task_index)
			soar_train_eval.monitored_sess(
				FLAGS=FLAGS,
				worker_count=worker_count, 
				task_index=FLAGS.task_index, 
				cluster=cluster, 
				is_chief=is_chief, 
				target=target,
				init_checkpoint=init_checkpoint,
				train_file=train_file,
				dev_file=dev_file,
				checkpoint_dir=checkpoint_dir,
				distribution_strategy="",
				rule_model=FLAGS.rule_model,
				parse_type=FLAGS.parse_type,
				train_op=FLAGS.train_op,
				running_type=FLAGS.running_type,
				input_target=FLAGS.input_target,
				decay=FLAGS.decay,
				warmup=FLAGS.warmup,
				distillation=FLAGS.distillation,
				temperature=FLAGS.temperature,
				distillation_ratio=FLAGS.distillation_ratio,
				sess_config=sess_config)

if __name__ == "__main__":
	tf.app.run()