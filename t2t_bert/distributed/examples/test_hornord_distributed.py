import sys,os
sys.path.append("../..")

import numpy as np
import tensorflow as tf
from example import hvd_distributed_classifier as bert_classifier
from bunch import Bunch
from example import feature_writer, write_to_tfrecords, classifier_processor
from data_generator import tokenization
# from data_generator import tf_data_utils
from data_generator import hvd_distributed_tf_data_utils as tf_data_utils
from model_io import model_io
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import horovod.tensorflow as hvd

from optimizer import hvd_distributed_optimizer as optimizer

def main(_):

	hvd.init()

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

	graph = tf.Graph()
	with graph.as_default():
		import json
		
		config = json.load(open("/data/xuht/bert/chinese_L-12_H-768_A-12/bert_config.json", "r"))
		init_checkpoint = "/data/xuht/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
		config = Bunch(config)
		config.use_one_hot_embeddings = True
		config.scope = "bert"
		config.dropout_prob = 0.1
		config.label_type = "single_label"
		config.loss = "focal_loss"
	#     config.num_hidden_layers = 
		
		# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

		num_train = int(33056 / hvd.size())

		batch_size = 32

		valid_step = int(num_train/batch_size)

		epoch = 2
		num_train_steps = int(
			num_train / (batch_size) * epoch)

		decay_train_steps = num_train_steps

		# decay_train_steps = int(
		# 		33056 / batch_size * epoch)

		num_warmup_steps = int(num_train_steps * 0.01)

		sess = tf.Session(config=sess_config)
		
		opt_config = Bunch({"init_lr":float(1e-5/hvd.size()), 
							"num_train_steps":decay_train_steps, 
							"cycle":False, 
							"num_warmup_steps":num_warmup_steps,
						   "lr_decay":"polynomial_decay"})
		model_io_config = Bunch({"fix_lm":False})
		
		model_io_fn = model_io.ModelIO(model_io_config)

		optimizer_fn = optimizer.Optimizer(opt_config)
		
		num_calsses = 2

		model_train_fn = bert_classifier.classifier_model_fn_builder(config, num_calsses, init_checkpoint, 
												reuse=None, 
												load_pretrained=True,
												model_io_fn=model_io_fn,
												optimizer_fn=optimizer_fn,
												model_io_config=model_io_config, 
												opt_config=opt_config)
		
		model_eval_fn = bert_classifier.classifier_model_fn_builder(config, num_calsses, init_checkpoint, 
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
						tf.FixedLenFeature([128], tf.int64),
				"input_mask":
						tf.FixedLenFeature([128], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([128], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
		}

		params = Bunch({})
		params.epoch = epoch
		params.batch_size = 32
		train_file = "/data/xuht/eventy_detection/event/model/train.tfrecords"
		train_file1 = "/data/xuht/eventy_detection/sentiment/model/sentiment_11_14/train.tfrecords"
		title_sentiment = "/data/xuht/eventy_detection/sentiment/model/test/train.tfrecords"
		sentiment = "/data/xuht/eventy_detection/sentiment/model/bert/train_11_15.tfrecords"
		jd_train = "/data/xuht/jd_comment/train.tfrecords"
		train_features = tf_data_utils.train_input_fn(jd_train,
									tf_data_utils._decode_record, name_to_features, params)
		
		test_file = ["/data/xuht/eventy_detection/sentiment/model/sentiment_11_14/test.tfrecords"]
		test_file1_1 = ["/data/xuht/eventy_detection/sentiment/model/test/train.tfrecords",
					"/data/xuht/eventy_detection/sentiment/model/test/test.tfrecords"]
		test_file2 = "/data/xuht/eventy_detection/event/model/test.tfrecords"
		title_test = "/data/xuht/eventy_detection/sentiment/model/test/test.tfrecords"
		jd_test = "/data/xuht/jd_comment/test.tfrecords"
		sentiment_test = "/data/xuht/eventy_detection/sentiment/model/bert/test_11_15.tfrecords"
		
		eval_features = tf_data_utils.eval_input_fn(jd_test,
									tf_data_utils._decode_record, name_to_features, params)
		
		[train_op, train_loss, train_per_example_loss, train_logits] = model_train_fn(train_features, [], tf.estimator.ModeKeys.TRAIN)
		[_, eval_loss, eval_per_example_loss, eval_logits] = model_eval_fn(eval_features, [], tf.estimator.ModeKeys.EVAL)
		result = metric_fn(eval_features, eval_logits, eval_loss)
		
		init_op = tf.group(tf.global_variables_initializer(), 
					tf.local_variables_initializer())
		sess.run(init_op)

		sess.run(hvd.broadcast_global_variables(0))
		
		model_io_fn.set_saver()

		print("===horovod rank==={}".format(hvd.rank()))
		
		def eval_fn(result):
			i = 0
			total_accuracy = 0
			label, label_id = [], []
			while True:
				try:
					eval_result = sess.run(result)
					total_accuracy += eval_result["accuracy"]
					label_id.extend(eval_result["label_ids"])
					label.extend(eval_result["pred_label"])
					i += 1
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break
			macro_f1 = f1_score(label_id, label, average="macro")
			micro_f1 = f1_score(label_id, label, average="micro")
			accuracy = accuracy_score(label_id, label)
			print("test accuracy {} macro_f1 score {} micro_f1 {} accuracy {}".format(total_accuracy/ i, 
																					macro_f1,  micro_f1, accuracy))
			return total_accuracy/ i, label_id, label
		
		def train_fn(op, loss):
			i = 0
			total_loss = 0
			cnt = 0
			while True:
				try:
					[_, train_loss] = sess.run([op, loss])
					i += 1
					cnt += 1
					total_loss += train_loss
					# print("==device id {} global step {}".format(hvd.rank(), step))
					if np.mod(i, valid_step) == 0:
						print(total_loss/cnt)
						cnt = 0
						total_loss = 0
				except tf.errors.OutOfRangeError:
					print("End of dataset")
					break
		import time
		start = time.time()
		train_fn(train_op, train_loss)
		acc, true_label, pred_label = eval_fn(result)
		end = time.time()
		print("==total time {} numbers of devices {}".format(end - start, hvd.size()))
	#     model_io_fn.save_model(sess, "/data/xuht/eventy_detection/sentiment/model/bert/sentiment.ckpt")

if __name__ == "__main__":
	tf.app.run()