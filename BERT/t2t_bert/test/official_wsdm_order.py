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

from example import bert_order_classifier

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
	"config_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"init_checkpoint", None,
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

flags.DEFINE_string(
	"lang", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", 100,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_type", None,
	"Input TF example files (can be a glob or comma separated).")

graph = tf.Graph()
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
with graph.as_default():
	import json

	tokenizer = tokenization.FullTokenizer(
	  vocab_file=FLAGS.vocab_file, 
		do_lower_case=True)

	classifier_data_api = classifier_processor.PiarOrderProcessor()

	eval_examples = classifier_data_api.get_test_examples(FLAGS.eval_data_file,
														FLAGS.lang)

	print(eval_examples[0].guid)

	label_tensor = np.asarray([0.18987, 0.20253, 0.60759]).astype(np.float32)

	label_id = json.load(open(FLAGS.label_id, "r"))

	num_choice = 3
	max_seq_length = FLAGS.max_length

	write_to_tfrecords.convert_pair_order_classifier_examples_to_features(eval_examples,
															label_id["label2id"],
														   max_seq_length,
														   tokenizer,
														   FLAGS.output_file)

	config = json.load(open(FLAGS.config_file, "r"))
	init_checkpoint = FLAGS.init_checkpoint

	print("===init checkoutpoint==={}".format(init_checkpoint))

	config = Bunch(config)
	config.use_one_hot_embeddings = True
	config.scope = "bert"
	config.dropout_prob = 0.2
	config.label_type = "single_label"
	
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	sess = tf.Session()
	
	opt_config = Bunch({"init_lr":1e-5, "num_train_steps":80000})
	model_io_config = Bunch({"fix_lm":False})
	
	model_io_fn = model_io.ModelIO(model_io_config)

	if FLAGS.model_type == "original":
		model_function = bert_order_classifier.classifier_model_fn_builder
	elif FLAGS.model_type == "attn":
		model_function = bert_order_classifier.classifier_attn_model_fn_builder
	elif FLAGS.model_type == "orignal_nonlinear":
		model_function = bert_order_classifier.classifier_model_fn_builder_v1

	model_eval_fn = model_function(
								config, 
								num_choice, 
								init_checkpoint, 
								model_reuse=None, 
								load_pretrained=True,
								model_io_fn=model_io_fn,
								model_io_config=model_io_config, 
								opt_config=opt_config,
								input_name=["a", "b"],
								label_tensor=None)
	
	# def metric_fn(features, logits):
	#     print(logits.get_shape(), "===logits shape===")
	#     pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
	#     return {"pred_label":pred_label, "qas_id":features["qas_id"]}

	def metric_fn(features, logits):
		print(logits.get_shape(), "===logits shape===")
		pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
		prob = tf.exp(tf.nn.log_softmax(logits))
		return {"pred_label":pred_label, 
				"qas_id":features["qas_id"],
				"prob":prob}
	
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
			"qas_id":
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
	params.epoch = 2
	params.batch_size = 32

	eval_features = tf_data_utils.eval_input_fn(FLAGS.output_file,
								_decode_record, name_to_features, params)
	
	[_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
	result = metric_fn(eval_features, eval_logits)
	
	model_io_fn.set_saver()
	
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)
	
	# def eval_fn(result):
	# 	i = 0
	# 	pred_label, qas_id = [], []
	# 	while True:
	# 		try:
	# 			eval_result = sess.run(result)
	# 			pred_label.extend(eval_result["pred_label"])
	# 			qas_id.extend(eval_result["qas_id"])
	# 			i += 1
	# 		except tf.errors.OutOfRangeError:
	# 			print("End of dataset")
	# 			break
	# 	return pred_label, qas_id

	def eval_fn(result):
		i = 0
		pred_label, qas_id, prob = [], [], []
		while True:
			try:
				eval_result = sess.run(result)
				pred_label.extend(eval_result["pred_label"].tolist())
				qas_id.extend(eval_result["qas_id"].tolist())
				prob.extend(eval_result["prob"].tolist())
				i += 1
			except tf.errors.OutOfRangeError:
				print("End of dataset")
				break
		return pred_label, qas_id, prob
	
	print("===========begin to eval============")
	[pred_label, qas_id, prob] = eval_fn(result)
	result = dict(zip(qas_id, pred_label))

	print(FLAGS.result_file.split("."))
	tmp_output = FLAGS.result_file.split(".")[0] + ".json"
	print(tmp_output, "===temp output===")
	json.dump({"id":qas_id,
				"label":pred_label,
				"prob":prob},
				open(tmp_output, "w"))

	print(len(result), "=====valid result======")

	print(len(result), "=====valid result======")

	import pandas as pd
	df = pd.read_csv(FLAGS.eval_data_file)

	output = {}
	for index in range(df.shape[0]):
		output[df.loc[index]["id"]] = ""

	final_output = []

	cnt = 0
	for key in output:
		if key in result:
			final_output.append({"Id":key, 
				"Category":label_id["id2label"][str(result[key])]})
			cnt += 1
		else:
			final_output.append({"Id":key, "Category":"unrelated"})
	
	df_out = pd.DataFrame(final_output)
	df_out.to_csv(FLAGS.result_file)

	print(len(output), cnt, len(final_output), "======num of results from model==========")

if __name__ == "__main__":
	flags.mark_flag_as_required("eval_data_file")
	flags.mark_flag_as_required("output_file")
	flags.mark_flag_as_required("config_file")
	flags.mark_flag_as_required("init_checkpoint")
	flags.mark_flag_as_required("result_file")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("label_id")
	flags.mark_flag_as_required("max_length")
	tf.app.run()


