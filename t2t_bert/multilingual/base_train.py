import sys,os
sys.path.append("..")

import numpy as np
import tensorflow as tf
from example import hvd_distributed_classifier as bert_classifier
from bunch import Bunch
from data_generator import tokenization
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
from model_io import model_io
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import horovod.tensorflow as hvd

from optimizer import hvd_distributed_optimizer as optimizer

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

def main(_):

	hvd.init()

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

	graph = tf.Graph()
	with graph.as_default():
		import json
				
		config = json.load(open(FLAGS.config_file, "r"))
		init_checkpoint = FLAGS.init_checkpoint

		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		
		if FLAGS.if_shard == "0":
			train_size = FLAGS.train_size
			epoch = int(FLAGS.epoch / hvd.size())
		elif FLAGS.if_shard == "1":
			train_size = int(FLAGS.train_size/hvd.size())
			epoch = FLAGS.epoch

		init_lr = 2e-5

		label_dict = json.load(open(FLAGS.label_id))

		num_train_steps = int(
			train_size / FLAGS.batch_size * epoch)
		num_warmup_steps = int(num_train_steps * 0.1)

		num_storage_steps = int(train_size / FLAGS.batch_size)

		print(" model type {}".format(FLAGS.model_type))

		print(num_train_steps, num_warmup_steps, "=============")
		
		opt_config = Bunch({"init_lr":init_lr/hvd.size(), 
							"num_train_steps":num_train_steps,
							"num_warmup_steps":num_warmup_steps})

		sess = tf.Session(config=sess_config)

		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)

		optimizer_fn = optimizer.Optimizer(opt_config)
		
		num_classes = FLAGS.num_classes

		model_train_fn = bert_classifier.classifier_model_fn_builder(config, num_classes, init_checkpoint, 
												reuse=None, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config)
		
		model_eval_fn = bert_classifier.classifier_model_fn_builder(config, num_classes, init_checkpoint, 
												reuse=True, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config)
		
		def metric_fn(features, logits, loss):
			print(logits.get_shape(), "===logits shape===")
			pred_label = tf.argmax(logits, axis=-1, output_type=tf.int32)
			prob = tf.nn.softmax(logits)
			accuracy = correct = tf.equal(
				tf.cast(pred_label, tf.int32),
				tf.cast(features["label_ids"], tf.int32)
			)
			accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
			return {"accuracy":accuracy, "loss":loss, "pred_label":pred_label, "label_ids":features["label_ids"]}
		
		name_to_features = {
				"input_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
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

		train_features = tf_data_utils.train_input_fn(FLAGS.train_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard)
		eval_features = tf_data_utils.eval_input_fn(FLAGS.dev_file,
									_decode_record, name_to_features, params, if_shard=FLAGS.if_shard)
		
		[train_op, train_loss, train_per_example_loss, train_logits] = model_train_fn(train_features, [], tf.estimator.ModeKeys.TRAIN)
		train_dict = {"train_op":train_op,
					"train_loss":train_loss}
		[_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
		eval_dict = metric_fn(eval_features, eval_logits, eval_loss)
		
		init_op = tf.group(tf.global_variables_initializer(), 
					tf.local_variables_initializer())
		sess.run(init_op)

		sess.run(hvd.broadcast_global_variables(0))
		
		model_io_fn.set_saver()

		print("===horovod rank==={}".format(hvd.rank()))

		def run_eval(steps):
			import _pickle as pkl
			eval_features = tf_data_utils.eval_input_fn(
										FLAGS.dev_file,
										_decode_record, 
										name_to_features, params)
			[_, eval_loss, 
			eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
			eval_dict = metric_fn(eval_features, eval_logits, eval_loss)
			sess.run(tf.local_variables_initializer())
			eval_finial_dict = eval_fn(eval_dict)
			if hvd.rank() == 0:
				pkl.dump(eval_finial_dict, open(FLAGS.model_output+"/eval_dict_{}.pkl".format(steps), "wb"))
			return eval_finial_dict
		
		def eval_fn(result):
			i = 0
			total_accuracy = 0
			eval_total_dict = {}
			
			while True:
				try:
					eval_result = sess.run(result)
					for key in eval_result:
						if key not in eval_total_dict:
							if key in ["pred_label", "label_ids"]:
								eval_total_dict[key] = []
								eval_total_dict[key].extend(eval_result[key])
							if key in ["accuracy", "loss"]:
								eval_total_dict[key] = 0.0
								eval_total_dict[key] += eval_result[key]
						else:
							if key in ["pred_label", "label_ids"]:
								eval_total_dict[key].extend(eval_result[key])
							if key in ["accuracy", "loss"]:
								eval_total_dict[key] += eval_result[key]

					i += 1
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break

			label_id = eval_total_dict["label_ids"]
			pred_label = eval_total_dict["pred_label"]

			result = classification_report(label_id, pred_label, 
				target_names=list(label_dict["label2id"].keys()))

			print(result)
			eval_total_dict["classification_report"] = result
			return eval_total_dict

		def train_fn(op_dict):
			i = 0
			cnt = 0
			loss_dict = {}
			monitoring_train = []
			monitoring_eval = []
			while True:
				try:
					train_result = sess.run(op_dict)
					for key in train_result:
						if key == "train_op":
							continue
						else:
							if np.isnan(train_result[key]):
								print(train_loss, "get nan loss")
								break
							else:
								if key in loss_dict:
									loss_dict[key] += train_result[key]
								else:
									loss_dict[key] = train_result[key]
					
					i += 1
					cnt += 1
					
					if np.mod(i, 100) == 0:
						string = ""
						for key in loss_dict:
							tmp = key + " " + str(loss_dict[key]/cnt) + "\t"
							string += tmp
						print(string)
						monitoring_train.append(loss_dict)

						eval_finial_dict = run_eval(int(i/num_storage_steps))
						monitoring_eval.append(eval_finial_dict)

						for key in loss_dict:
							loss_dict[key] = 0.0
						if hvd.rank() == 0:
							model_io_fn.save_model(sess, FLAGS.model_output+"/model_{}.ckpt".format(int(i/num_storage_steps)))
							print("==successful storing model=={}".format(int(i/num_storage_steps)))
						cnt = 0

				except tf.errors.OutOfRangeError:
					if hvd.rank() == 0:
						import _pickle as pkl
						pkl.dump({"train":monitoring_train,
							"eval":monitoring_eval}, open(FLAGS.model_output+"/monitoring.pkl", "wb"))

					break
		print("===========begin to train============")        
		train_fn(train_dict)
		if hvd.rank() == 0:
			model_io_fn.save_model(sess, FLAGS.model_output+"/model.ckpt")
			print("===========begin to eval============")
			eval_finial_dict = run_eval("final")

if __name__ == "__main__":
	tf.app.run()