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

flags.DEFINE_string("buckets", "", "oss buckets")
flags.DEFINE_string("local_buckets", "", "oss buckets")

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_dir", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"multi_task_config", None,
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

def export_model(FLAGS,
				**kargs):

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	model_dir = os.path.join(FLAGS.buckets, FLAGS.model_dir)
	export_path = os.path.join(FLAGS.buckets, FLAGS.export_path)

	print(init_checkpoint, model_dir, export_path, "==load and store file on ==", FLAGS.buckets)

	distillation_config = Bunch(json.load(tf.gfile.Open(FLAGS.multi_task_config)))

	num_classes = FLAGS.num_classes

	model_config_dict = {}
	num_labels_dict = {}
	init_checkpoint_dict = {}
	load_pretrained_dict = {}
	exclude_scope_dict = {}
	not_storage_params_dict = {}
	target_dict = {}

	for task_type in FLAGS.multi_task_type.split(","):
		print("==task type==", task_type)
		model_config_dict[task_type] = model_config_parser(Bunch(distillation_config[task_type]))
		model_config_dict[task_type].update(distillation_config[task_type])
		print(task_type, distillation_config[task_type], '=====task model config======')
		num_labels_dict[task_type] = distillation_config[task_type]["num_labels"]
		init_checkpoint_dict[task_type] = os.path.join(FLAGS.buckets, distillation_config[task_type]["init_checkpoint"])
		load_pretrained_dict[task_type] = distillation_config[task_type]["load_pretrained"]
		exclude_scope_dict[task_type] = distillation_config[task_type]["exclude_scope"]
		not_storage_params_dict[task_type] = distillation_config[task_type]["not_storage_params"]
		target_dict[task_type] = distillation_config[task_type]["target"]

	def serving_input_receiver_fn():
		receiver_features = {
			"input_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids'),
			"input_mask":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_mask'),
			"segment_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='segment_ids'),
			"input_ori_ids":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ori_ids'),
			"next_sentence_labels":tf.placeholder(tf.int32, [None], name='next_sentence_labels')
		}
		print(receiver_features, "==input receiver_features==")
		input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
		return input_fn

	model_fn = electra_export.classifier_model_fn_builder(model_config_dict,
				num_labels_dict,
				init_checkpoint_dict,
				load_pretrained_dict,
				model_io_config=model_io_config,
				opt_config={},
				exclude_scope_dict=exclude_scope_dict,
				not_storage_params_dict=not_storage_params_dict,
				target_dict=target_dict,
				use_tpu=False,
				graph=None,
				num_train_steps=None,
				**kargs)

	estimator = tf.estimator.Estimator(
				model_fn=model_fn,
				model_dir=checkpoint_dir)

	export_dir = estimator.export_savedmodel(export_dir, 
									serving_input_receiver_fn,
									checkpoint_path=init_checkpoint)
	print("===Succeeded in exporting saved model==={}".format(export_dir))
