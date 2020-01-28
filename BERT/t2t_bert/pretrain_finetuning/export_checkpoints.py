# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, json, re
import tensorflow as tf
import logging

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

flags = tf.flags

import numpy as np
from pretrain_finetuning.generator_exporter import model_fn_builder as generator_model_fn
from pretrain_finetuning.discriminator_exporter import model_fn_builder as discriminator_model_fn
from bunch import Bunch
import collections
from model_io import model_io

flags.DEFINE_string(
		"buckets", None,
		"The config json file corresponding to the pre-trained ALBERT model. "
		"This specifies the model architecture.")

flags.DEFINE_string(
		"config_path", None,
		"The config json file corresponding to the pre-trained ALBERT model. "
		"This specifies the model architecture.")

flags.DEFINE_string(
		"checkpoint_path", "model.ckpt-best",
		"Name of the checkpoint under albert_directory to be exported.")

flags.DEFINE_string(
		"model_type", 'bert',
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_string(
		"electra", 'generator',
		"if discriminator, we export discriminator or we export generator "
		"models and False for cased models.")

flags.DEFINE_string(
		"model_scope", 'bert',
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_string(
		"exclude_scope", 'generator',
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_string("export_path", None, "Path to the output module.")
flags.DEFINE_string("sharing_mode", "all_sharing", "Path to the output module.")
flags.DEFINE_string("ln_type", "postln", "Path to the output module.")

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.logging.set_verbosity(tf.logging.INFO)

father_path = os.path.join(os.getcwd())
print(father_path, "==father path==")

def input_fn():
	if FLAGS.electra == 'generator':
		features = {
			"input_ids":tf.placeholder(tf.int32, [32, 128], "input_ids"),
			"input_mask":tf.placeholder(tf.int32, [32, 128], "input_mask"),
			"segment_ids":tf.placeholder(tf.int32, [32, 128], "segment_ids"),
			"next_sentence_labels":tf.placeholder(tf.int32, [32], "next_sentence_labels"),
			"input_ori_ids":tf.placeholder(tf.int32, [32, 128], "input_ori_ids"),
		}
	elif FLAGS.electra == 'discriminator':
		features = {
			"input_ids":tf.placeholder(tf.int32, [32, 128], "input_ids"),
			"input_mask":tf.placeholder(tf.int32, [32, 128], "input_mask"),
			"segment_ids":tf.placeholder(tf.int32, [32, 128], "segment_ids"),
			"next_sentence_labels":tf.placeholder(tf.int32, [32], "next_sentence_labels"),
			"input_ori_ids":tf.placeholder(tf.int32, [32, 128], "input_ori_ids"),
		}
	else:
		features = {
			"input_ids":tf.placeholder(tf.int32, [32, 128], "input_ids"),
			"input_mask":tf.placeholder(tf.int32, [32, 128], "input_mask"),
			"segment_ids":tf.placeholder(tf.int32, [32, 128], "segment_ids"),
			"next_sentence_labels":tf.placeholder(tf.int32, [], "next_sentence_labels"),
			"input_ori_ids":tf.placeholder(tf.int32, [32, 128], "input_ori_ids"),
		}
	return features

def remove_exclude_scope(name, exclude_scope):
	if_exclude = re.search(exclude_scope+"/", name)
	if if_exclude:
		return re.sub(exclude_scope+"/", "", name)
	else:
		return None

def get_assigment_map_from_checkpoint(tvars, init_checkpoint, **kargs):
	"""Compute the union of the current variables and checkpoint variables."""
	assignment_map = {}
	initialized_variable_names = {}

	exclude_scope = kargs.get("exclude_scope", "")

	name_to_variable = collections.OrderedDict()
	for var in tvars:
		name = var.name
		m = re.match("^(.*):\\d+$", name)
		if m is not None:
			name = m.group(1)
		name_to_variable[name] = var

	init_vars = tf.train.list_variables(init_checkpoint)
	init_vars_name_list = []

	assignment_map = collections.OrderedDict()
	for x in init_vars:
		(name, var) = (x[0], x[1])
		init_vars_name_list.append(name)
		if len(exclude_scope) >= 1:
			assignment_name = remove_exclude_scope(name, exclude_scope)
		else:
			assignment_name = name

		if len(exclude_scope):
			if not re.search(exclude_scope, name):
				continue

		if assignment_name not in name_to_variable or not assignment_name:
			continue
		assignment_map[name] = assignment_name
		initialized_variable_names[assignment_name] = 1
		initialized_variable_names[assignment_name + ":0"] = 1

	flag = 1

	for name in name_to_variable:
		if name not in initialized_variable_names and name in init_vars_name_list:
			assignment_map[name] = name
			initialized_variable_names[name] = 1
			initialized_variable_names[name + ":0"] = 1
			flag = 0
			tf.logging.info("***** restore: %s from checkpoint ******", name)
	if flag == 1:
		tf.logging.info("***** no need extra restoring variables from checkpoint ******")

	return (assignment_map, initialized_variable_names)

def build_model(sess):
	"""Module function."""
	features = input_fn()

	bert_config_path = FLAGS.config_path
	bert_config = Bunch(json.load(open(bert_config_path)))
	bert_config.scope = FLAGS.model_scope
	bert_config.dropout_prob = 0.1

	bert_config.use_one_hot_embeddings = True
	bert_config.label_type = "single_label"
	bert_config.model_type = FLAGS.model_type
	bert_config.ln_type = FLAGS.ln_type
	bert_config.lm_ratio = 0.1

	model_io_fn = model_io.ModelIO({"fix_lm":False})

	if FLAGS.electra == 'generator':
		model_fn = generator_model_fn(bert_config,
					2,
					init_checkpoint=None,
					model_reuse=None,
					load_pretrained="no",
					model_io_config=Bunch({"fix_lm":False}),
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="",
					sharing_mode=FLAGS.sharing_mode)
	elif FLAGS.electra == 'discriminator':
		model_fn = discriminator_model_fn(bert_config,
					2,
					init_checkpoint=None,
					model_reuse=None,
					load_pretrained=True,
					model_io_config=Bunch({"fix_lm":False}),
					opt_config={},
					exclude_scope="",
					not_storage_params=[],
					target="",
					sharing_mode=FLAGS.sharing_mode)

	model_fn(features, [], tf.estimator.ModeKeys.TRAIN, {})

	checkpoint_path = os.path.join(FLAGS.buckets, FLAGS.checkpoint_path)
	tvars = tf.trainable_variables()

	(assignment_map, initialized_variable_names
	) = get_assigment_map_from_checkpoint(tvars, checkpoint_path, exclude_scope=FLAGS.exclude_scope)

	tf.logging.info("**** Trainable Variables ****")
	for var in tvars:
		init_string = ""
		if var.name in initialized_variable_names:
			init_string = ", *INIT_FROM_CKPT*"
		tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
										init_string)
	tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
	init = tf.global_variables_initializer()
	sess.run(init)
	return sess, tvars


def main():
	sess = tf.Session()
	tf.train.get_or_create_global_step()
	sess, my_vars = build_model(sess)
	saver = tf.train.Saver(my_vars)
	saver.save(sess, os.path.join(FLAGS.buckets, FLAGS.export_path))

if __name__ == "__main__":
	main()