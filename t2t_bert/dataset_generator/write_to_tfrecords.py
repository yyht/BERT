import sys,os,json

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

from dataset_generator.create_generator import create_generator
import tensorflow as tf
import json, os, sys
from data_generator import tf_data_utils
from bunch import Bunch

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

def write2tfrecords():
	multi_task_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))
	generator = create_generator(FLAGS, multi_task_config, "train", FLAGS.epoch)

	_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.buckets, FLAGS.model_output))
	problem_config = multi_task_config[FLAGS.multi_task_type.split(",")[0]]

	for idx, item in enumerate(generator):
		features = {}
		features["input_ids"] = tf_data_utils.create_int_feature(item["input_ids"])
		features["input_mask"] = tf_data_utils.create_int_feature(item["input_mask"])
		features["segment_ids"] = tf_data_utils.create_int_feature(item["segment_ids"])

		if problem_config["lm_augumentation"]:
			features["masked_lm_positions"] = tf_data_utils.create_int_feature(item["masked_lm_positions"])
			features["masked_lm_ids"] = tf_data_utils.create_int_feature(item["masked_lm_ids"])
			features["masked_lm_weights"] = tf_data_utils.create_int_feature(item["masked_lm_weights"])

		for problem in FLAGS.multi_task_type.split(","):
			problem_dict = multi_task_config[problem]
			problem_type = multi_task_config[problem]["task_type"]

			features["{}_loss_multiplier".format(problem)] = tf_data_utils.create_int_feature([item["{}_loss_multiplier".format(problem)]])
			if problem_type in ['cls_task']:
				features["{}_label_ids".format(problem)] = tf_data_utils.create_int_feature([item["{}_label_ids".format(problem)]])
			elif problem_type in ['seq2seq_tag_task', 'seq2seq_text_task']:
				features["{}_label_ids".format(problem)] = tf_data_utils.create_int_feature(item["{}_label_ids".format(problem)])
			
			features["task_id"] = tf_data_utils.create_int_feature([item["task_id"]])

		try:
			features["guid"] = tf_data_utils.create_int_feature([idx])
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			_writer.write(tf_example.SerializeToString())
		except:
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			_writer.write(tf_example.SerializeToString())
		break

write2tfrecords()



	





