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
from pretrain_finetuning import electra_export

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

flags.DEFINE_integer("save_checkpoints_steps", 1000,
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

graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = model_config_parser(FLAGS)
		
		train_size = int(FLAGS.train_size)
		init_lr = FLAGS.init_lr

		distillation_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))
