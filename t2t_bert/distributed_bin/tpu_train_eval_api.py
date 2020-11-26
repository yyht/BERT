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
from pretrain_finetuning import train_eval_tpu_estimator
from pretrain_finetuning import train_eval_gpu_electra_estimator


flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

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

flags.DEFINE_string(
	"label_type", "single_label",
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

flags.DEFINE_string(
	"distribution_strategy", "ParameterServerStrategy",
	"distribution strategy"
	)

flags.DEFINE_string(
	"rule_model", "normal",
	"distribution strategy"
	)

flags.DEFINE_string(
	"parse_type", "parse_single", 
	"the required num_gpus"
	)

flags.DEFINE_string(
	"profiler", "normal", 
	"the required num_gpus"
	)


flags.DEFINE_string(
	"train_op", "adam_decay", 
	"the required num_gpus"
	)

flags.DEFINE_integer(
	"num_gpus", 4, 
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
	"apply_cpc", 'none',
	"if apply distillation"
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

flags.DEFINE_float(
	"init_lr", 5e-5,
	"if apply distillation"
	)

flags.DEFINE_integer(
	"max_predictions_per_seq", 10,
	"if apply distillation"
	)

flags.DEFINE_string(
			"ln_type", 'postln',
				"if apply distillation"
					)


flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
					 "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
					 "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
	"tpu_name", None,
	"The Cloud TPU to use for training. This should be either the name "
	"used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
	"url.")

tf.flags.DEFINE_string(
	"tpu_zone", None,
	"[Optional] GCE zone where the Cloud TPU is located in. If not "
	"specified, we will attempt to automatically detect the GCE project from "
	"metadata.")

tf.flags.DEFINE_string(
	"gcp_project", None,
	"[Optional] Project name for the Cloud TPU-enabled project. If not "
	"specified, we will attempt to automatically detect the GCE project from "
	"metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
	"num_tpu_cores", 8,
	"Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
	"joint_train", "0",
	"if apply distillation"
	)

flags.DEFINE_string(
	"optimization_type", "grl",
	"if apply distillation"
	)

flags.DEFINE_string(
	"train_op_type", "joint",
	"if apply distillation"
	)

flags.DEFINE_string(
	"random_generator", "1",
	"if apply distillation"
	)

flags.DEFINE_string(
	"electra_mode", "solo_training",
	"if apply distillation"
	)

flags.DEFINE_string(
	"sharing_mode", "none",
	"if apply distillation"
	)

flags.DEFINE_string(
	"attention_type", "normal_attention",
	"if apply distillation"
	)

flags.DEFINE_string(
	"gumbel_anneal", "anneal",
	"if apply distillation"
	)

flags.DEFINE_string(
	"exclude_scope", "",
	"if apply distillation"
	)

flags.DEFINE_bool(
	"annealed_mask_prob", False,
	"if apply distillation"
	)

flags.DEFINE_string(
	"model_scope", "bert",
	"if apply distillation"
	)

flags.DEFINE_string(
	"gen_disc_type", "all_disc",
	"if apply distillation"
	)

flags.DEFINE_string(
	"mask_method", "only_mask",
	"if apply distillation"
	)

flags.DEFINE_string(
	"minmax_mode", "corrupted",
	"if apply distillation"
	)

flags.DEFINE_string(
	"seq_type", "none",
	"if apply distillation"
	)

flags.DEFINE_string(
	"mask_type", "none",
	"if apply distillation"
	)

flags.DEFINE_string(
	"confusion_set_path", "none",
	"if apply distillation"
	)

flags.DEFINE_string(
	"confusion_set_mask_path", "none",
	"if apply distillation"
	)

flags.DEFINE_string(
	"train_file_path", "none",
	"if apply distillation"
	)

import random
def main(_):

	tf.enable_resource_variables()

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	train_file = []
	# try:
	with tf.gfile.GFile(os.path.join(FLAGS.buckets, FLAGS.train_file_path), "r") as reader:
		for index, line in enumerate(reader):
			content = line.strip()
			train_file_path = os.path.join(FLAGS.buckets, content)
			train_file.append(train_file_path)
	print(train_file)
	# train_file = [train_file[0]]
	# except:
	# 	for file in FLAGS.train_file.split(","):
	# 		train_file_path = os.path.join(FLAGS.buckets, file)
	# 		train_file.append(train_file_path)
	# 	print(train_file_path)
	random.shuffle(train_file)

	tf.logging.info("** total data file:%s **"%(str(len(train_file))))

	dev_file = []
	for file in FLAGS.dev_file.split(","):
		dev_file_path = os.path.join(FLAGS.buckets, file)
		dev_file.append(dev_file_path)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)

	print(init_checkpoint, train_file, dev_file, checkpoint_dir)

	tpu_cluster_resolver = None
	if FLAGS.use_tpu and FLAGS.tpu_name:
		tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver( # TODO
			tpu=FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

	print("###tpu_cluster_resolver:",tpu_cluster_resolver,";FLAGS.use_tpu:",FLAGS.use_tpu,";FLAGS.tpu_name:",FLAGS.tpu_name,";FLAGS.tpu_zone:",FLAGS.tpu_zone)
	# ###tpu_cluster_resolver: <tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver.TPUClusterResolver object at 0x7f4b387b06a0> ;FLAGS.use_tpu: True ;FLAGS.tpu_name: grpc://10.240.1.83:8470

	tf.logging.info("****** tpu_name ******* %s", FLAGS.tpu_name)

	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
	  keep_checkpoint_max=100, # 10
	  cluster=tpu_cluster_resolver,
	  master=FLAGS.master,
	  model_dir=checkpoint_dir,
	  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
	  tpu_config=tf.contrib.tpu.TPUConfig(
		  iterations_per_loop=FLAGS.iterations_per_loop,
		  num_shards=FLAGS.num_tpu_cores,
		  per_host_input_for_training=is_per_host))
	print(FLAGS.do_train, "=====do train flag======")

	if FLAGS.mode == 'pretrain':
		train_eval_tpu_estimator.train_eval_fn(FLAGS=FLAGS,
			init_checkpoint=init_checkpoint,
			train_file=train_file,
			dev_file=dev_file,
			checkpoint_dir=checkpoint_dir,
			run_config=run_config,
			train_op=FLAGS.train_op,
			decay=FLAGS.decay,
			warmup=FLAGS.warmup,
			input_target=FLAGS.input_target,
			attention_type=FLAGS.attention_type,
			exclude_scope=FLAGS.exclude_scope,
			annealed_mask_prob=FLAGS.annealed_mask_prob,
			seq_type=FLAGS.seq_type,
			mask_type=FLAGS.mask_type)
	elif FLAGS.mode == 'electra':
		train_eval_gpu_electra_estimator.train_eval_fn(
			FLAGS=FLAGS,
			init_checkpoint=init_checkpoint,
			train_file=train_file,
			dev_file=dev_file,
			checkpoint_dir=checkpoint_dir,
			run_config=run_config,
			train_op=FLAGS.train_op,
			decay=FLAGS.decay,
			warmup=FLAGS.warmup,
			input_target=FLAGS.input_target,
			electra_mode=FLAGS.electra_mode,
			joint_train=FLAGS.joint_train,
			sharing_mode=FLAGS.sharing_mode,
			attention_type=FLAGS.attention_type,
			optimization_type=FLAGS.optimization_type,
			train_op_type=FLAGS.train_op_type,
			gumbel_anneal=FLAGS.gumbel_anneal,
			# exclude_scope=FLAGS.exclude_scope,
			annealed_mask_prob=FLAGS.annealed_mask_prob,
			gen_disc_type=FLAGS.gen_disc_type,
			mask_method=FLAGS.mask_method,
			minmax_mode=FLAGS.minmax_mode,
			seq_type=FLAGS.seq_type,
			mask_type=FLAGS.mask_type)


if __name__ == "__main__":
		tf.app.run()
