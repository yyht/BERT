import sys,os
sys.path.append("..")
from model_io import model_io
import numpy as np
import tensorflow as tf
from example import bert_classifier
from bunch import Bunch

from example import feature_writer, write_to_tfrecords, classifier_processor
from data_generator import tokenization
from data_generator import tf_data_utils

from knowledge_distillation import distillation

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"eval_data_file", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"output_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"student_config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"student_init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"teacher_init_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"result_file", None,
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

flags.DEFINE_string(
	"gpu_id", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"epoch", 5,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"num_classes", 3,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"train_size", 256434,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"batch_size", 16,
	"Input TF example files (can be a glob or comma separated).")


def main(_):
	graph = tf.Graph()
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	with graph.as_default():
		import json

		os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
		sess = tf.Session()

		config = json.load(open(FLAGS.config_file, "r"))

		student_config = json.load(open(FLAGS.student_config_file, "r"))

		student_config = Bunch(student_config)
		# student_config.use_one_hot_embeddings = True
		# student_config.scope = "student/bert"
		# student_config.dropout_prob = 0.1
		# student_config.label_type = "single_label"
		# student_config.init_checkpoint = FLAGS.student_init_checkpoint

		temperature = student_config.temperature
		distill_ratio = student_config.distill_ratio

		# json.dump(student_config, open(FLAGS.model_output+"/student_config.json", "w"))

		teacher_config = Bunch(config)
		teacher_config.use_one_hot_embeddings = True
		teacher_config.scope = "teacher/bert"
		teacher_config.dropout_prob = 0.1
		teacher_config.label_type = "single_label"
		teacher_config.init_checkpoint = FLAGS.teacher_init_checkpoint

		# json.dump(teacher_config, open(FLAGS.model_output+"/teacher_config.json", "w"))

		model_config_dict = {"student":student_config, "teacher":teacher_config}
		init_checkpoint_dict = {"student":FLAGS.student_init_checkpoint,
							   "teacher":FLAGS.teacher_init_checkpoint}

		print("==student checkpoint=={}".format(FLAGS.student_init_checkpoint))

		num_train_steps = int(
			FLAGS.train_size / FLAGS.batch_size * FLAGS.epoch)
		num_warmup_steps = int(num_train_steps * 0.1)

		num_storage_steps = int(FLAGS.train_size / FLAGS.batch_size)

		print(num_train_steps, num_warmup_steps, "=============")
		
		opt_config = Bunch({"init_lr":1e-5, 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps})

		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)
		
		num_choice = FLAGS.num_classes
		max_seq_length = FLAGS.max_length

		model_eval_fn = distillation.distillation_model_fn(
			model_config_dict=model_config_dict,
			num_labels=num_choice,
			init_checkpoint_dict=init_checkpoint_dict,
			model_reuse=None,
			load_pretrained={"teacher":True, "student":True},
			model_io_fn=model_io_fn,
			model_io_config=model_io_config,
			opt_config=opt_config,
			student_input_name=["a", "b"],
			teacher_input_name=["a", "b"],
			unlabel_input_name=["ua", "ub"],
			temperature=temperature,
			exclude_scope_dict={"student":"", "teacher":"teacher"},
			not_storage_params=["adam_m", "adam_v"],
			distillation_weight={"label":distill_ratio, "unlabel":distill_ratio},
			if_distill_unlabeled=False
		)

		def metric_fn(features, logits, loss):
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			accuracy = correct = tf.equal(
				tf.cast(pred_label, tf.int32),
				tf.cast(features["label_ids"], tf.int32)
			)
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
			return {"accuracy":accuracy, "loss":loss, 
					"pred_label":pred_label, "label_ids":features["label_ids"],
					"pred_prob":prob}
					
		name_to_features = {
				"input_ids_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids_a":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_ids_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids_b":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
		}
		
		def _decode_record(record, name_to_features):
			"""Decodes a record to a TensorFlow example.
			"""
			example = tf.parse_single_example(record, name_to_features)

			# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
			# So cast all int64 to int32.
			for name in list(example.keys()):
				t = example[name]
				if t.dtype == tf.int64:
					t = tf.to_int32(t)
				example[name] = t
			return example

		params = Bunch({})
		params.epoch = FLAGS.epoch
		params.batch_size = FLAGS.batch_size
		# train_features = tf_data_utils.train_input_fn("/data/xuht/wsdm19/data/train.tfrecords",
		#                             _decode_record, name_to_features, params)
		# eval_features = tf_data_utils.eval_input_fn("/data/xuht/wsdm19/data/dev.tfrecords",
		#                             _decode_record, name_to_features, params)

		eval_features = tf_data_utils.eval_input_fn(FLAGS.dev_file,
									_decode_record, name_to_features, params)

		[_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
		result = metric_fn(eval_features, eval_logits, eval_loss)
		
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		
		def eval_fn(result):
			i = 0
			total_accuracy = 0
			total_loss = 0.0
			pred_prob = []
			label, label_id = [], []
			while True:
				try:
					eval_result = sess.run(result)
					total_accuracy += eval_result["accuracy"]
					label_id.extend(eval_result["label_ids"])
					label.extend(eval_result["pred_label"])
					total_loss += eval_result["loss"]
					pred_prob.extend(eval_result["pred_prob"])
					i += 1
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break
			f1 = f1_score(label_id, label, average="macro")
			accuracy = accuracy_score(label_id, label)
			print("test accuracy {} accuracy {} loss {} f1 {}".format(total_accuracy/i, 
				accuracy, total_loss/i, f1))
			return accuracy, f1, pred_prob
		
		print("===========begin to eval============")
		accuracy, f1, label = eval_fn(result)
		print("==accuracy {} f1 {} size {}==".format(accuracy, f1, len(label)))
		# model_io_fn.save_model(sess, "/data/xuht/wsdm19/data/model_11_15_focal_loss/oqmrc.ckpt")


if __name__ == "__main__":
	flags.mark_flag_as_required("eval_data_file")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("config_file")
	flags.mark_flag_as_required("student_init_checkpoint")
	flags.mark_flag_as_required("teacher_init_checkpoint")
	flags.mark_flag_as_required("result_file")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("train_file")
	flags.mark_flag_as_required("dev_file")
	flags.mark_flag_as_required("max_length")
	flags.mark_flag_as_required("model_output")
	flags.mark_flag_as_required("gpu_id")
	flags.mark_flag_as_required("epoch")
	flags.mark_flag_as_required("num_classes")
	tf.app.run()                    