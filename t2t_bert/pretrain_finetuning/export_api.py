
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
from pretrain_finetuning import export_discriminator

import tensorflow as tf
import json

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
	"model_type", "bert",
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

flags.DEFINE_string(
	"input_target", "a", 
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
	"export_dir", "",
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

flags.DEFINE_string(
	"distillation_config", 'postln',
	"if apply distillation"
	)

flags.DEFINE_bool(
	"use_tpu", False,
	"if apply distillation"
	)

flags.DEFINE_string(
	"joint_train", "0",
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
	"exclude_scope", "",
	"if apply distillation"
	)

flags.DEFINE_string(
	"ues_token_type", "yes",
	"if apply distillation"
	)

flags.DEFINE_string(
	"model_scope", "bert",
	"if apply distillation"
	)

flags.DEFINE_string(
	"gumbel_anneal", "anneal",
	"if apply distillation"
	)

flags.DEFINE_bool(
	"annealed_mask_prob", False,
	"if apply distillation"
	)

flags.DEFINE_string(
	"optimization_type", "grl",
	"if apply distillation"
	)

flags.DEFINE_string(
	"gen_disc_type", "all_disc",
	"if apply distillation"
	)


flags.DEFINE_string(
	"train_op_type", "joint",
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
	"export_type", "discriminator",
	"if apply distillation"
	)

def main(_):

	print(FLAGS)
	print(tf.__version__, "==tensorflow version==")

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)
	export_dir = os.path.join(FLAGS.buckets, FLAGS.export_dir)

	print(init_checkpoint, checkpoint_dir, export_dir)

	export_discriminator.export_model(FLAGS,
						init_checkpoint,
						checkpoint_dir,
						export_dir,
						input_target=FLAGS.input_target,
						export_type=FLAGS.export_type,
						sharing_mode=FLAGS.sharing_mode)

if __name__ == "__main__":
	tf.app.run()