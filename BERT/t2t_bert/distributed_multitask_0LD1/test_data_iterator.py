import tensorflow as tf
from bunch import Bunch
import os
import numpy as np

flags = tf.flags
os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

## Required parameters
flags.DEFINE_string(
	"train_file", "",
	"Input TF example files (can be a glob or comma separated).")

def test():
	name_to_features = {
			"input_ids":tf.FixedLenFeature([128], tf.int64),
			"input_mask":tf.FixedLenFeature([128], tf.int64),
			"wsdm_label_ids":tf.FixedLenFeature([], tf.int64),
			"wsdm_mask":tf.FixedLenFeature([], tf.int64),
			"segment_ids":tf.FixedLenFeature([128], tf.int64),
			"chnsenticorp_label_ids":tf.FixedLenFeature([], tf.int64),
			"chnsenticorp_mask":tf.FixedLenFeature([], tf.int64),
			"xnli_mask":tf.FixedLenFeature([], tf.int64),
			"xnli_label_ids":tf.FixedLenFeature([], tf.int64),
			"nlpcc-dbqa_label_ids":tf.FixedLenFeature([], tf.int64),
			"nlpcc-dbqa_mask":tf.FixedLenFeature([], tf.int64),
			"jd_comment_label_ids":tf.FixedLenFeature([], tf.int64),
			"jd_comment_mask":tf.FixedLenFeature([], tf.int64),
			"ant_label_ids":tf.FixedLenFeature([], tf.int64),
			"ant_mask":tf.FixedLenFeature([], tf.int64),
		}


	name_to_features = {'segment_ids': tf.FixedLenFeature(shape=[128], dtype=tf.int64, default_value=None), 
 'lcqmc_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'ccks_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'lcqmc_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'jd_comment_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'nlpcc-dbqa_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'xnli_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'wsdm_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'nlpcc-dbqa_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'xnli_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'input_ids': tf.FixedLenFeature(shape=[128], dtype=tf.int64, default_value=None), 
 'ant_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'chnsenticorp_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'ant_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'wsdm_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'ccks_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'jd_comment_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'chnsenticorp_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None)
}

	params = Bunch({})
	params.epoch = 1
	params.batch_size = 24

	def _decode_record(record, name_to_features):
		example = tf.parse_example(record, name_to_features)
		return example

	def train_input_fn(input_file, _parse_fn, name_to_features,
		params, **kargs):
		if_shard = kargs.get("if_shard", "1")

		worker_count = kargs.get("worker_count", 1)
		task_index = kargs.get("task_index", 0)

		dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
		print("==aaworker_count {}, task_index {}==".format(worker_count, task_index))
		if if_shard == "1":
			dataset = dataset.shard(worker_count, task_index)
		dataset = dataset.shuffle(
								buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
								seed=np.random.randint(0,1e10,1)[0],
								reshuffle_each_iteration=True)
		dataset = dataset.batch(params.get("batch_size", 32))
		dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
		
		
		dataset = dataset.repeat(1)
		iterator = dataset.make_one_shot_iterator()
		features = iterator.get_next()
		return features

	def train_batch_input_fn(input_file, _parse_fn, name_to_features,
		params, **kargs):
		if_shard = kargs.get("if_shard", "1")

		worker_count = kargs.get("worker_count", 1)
		task_index = kargs.get("task_index", 0)

		# dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))

		dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_file))
		# dataset = dataset.shuffle(buffer_size=len(input_file))
		dataset = dataset.repeat(params.get("epoch", 100))

		# `cycle_length` is the number of parallel files that get read.
		cycle_length = min(4, len(input_file))

		# `sloppy` mode means that the interleaving is not exact. This adds
		# even more randomness to the training pipeline.
		dataset = dataset.apply(
					tf.contrib.data.parallel_interleave(
					  tf.data.TFRecordDataset,
					  sloppy=True,
					  cycle_length=cycle_length))

		print("==worker_count {}, task_index {}==".format(worker_count, task_index))
		if if_shard == "1":
			dataset = dataset.shard(worker_count, task_index)
		dataset = dataset.shuffle(
								buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
								seed=np.random.randint(0,1e10,1)[0],
								reshuffle_each_iteration=True)
		dataset = dataset.batch(params.get("batch_size", 32))
		dataset = dataset.map(lambda x:_parse_fn(x, name_to_features),
							num_parallel_calls=kargs.get("num_parallel_calls", 10))
		iterator = dataset.make_one_shot_iterator()
		features = iterator.get_next()
		return features

	# file_test = "/data/xuht/porn/clean_data/textcnn/distillation/train_tfrecords"



	input_fn = train_batch_input_fn([
								"/data/xuht/multi_task/data/single_sentence/chnsenticorp/train_tfrecords",
								"/data/xuht/multi_task/data/sentence_pair/xnli/train_tfrecords",
								"/data/xuht/multi_task/data/sentence_pair/lcqmc/train_tfrecords",
								"/data/xuht/multi_task/data/sentence_pair/ccks/train_tfrecords",
								"/data/xuht/multi_task/data/sentence_pair/wsdm/train_tfrecords",
								"/data/xuht/multi_task/data/sentence_pair/ant/train_tfrecords",
								"/data/xuht/multi_task/data/qa/nlpcc-dbqa/train_tfrecords"],
						_decode_record, name_to_features, params)
	
	sess = tf.Session()
	
	init_op = tf.group(
				tf.local_variables_initializer())
	sess.run(init_op)
	
	i = 0
	cnt = 0
	f = {}
	
	while True:
		try:
			features = sess.run(input_fn)
			mask_num = 0
			for key in features:
				if "mask" in key:
					if key in f:
						f[key].extend(features[key].tolist())
					else:
						f[key] = features[key].tolist()
					mask_num += np.sum(features[key])
				print(features[key], key)
			print(mask_num, type(mask_num))
			if mask_num == 0:
				break
		except tf.errors.OutOfRangeError:
			print("End of dataset")
			break

test()