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
			"ant_label_ids":tf.FixedLenFeature([], tf.int64),
			"ant_mask":tf.FixedLenFeature([], tf.int64),
		}


	name_to_features = {'segment_ids': tf.FixedLenFeature(shape=[256], dtype=tf.int64, default_value=None), 
 'lcqmc_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'ccks_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 'lcqmc_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'jd_comment_mask': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'nlpcc-dbqa_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'xnli_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'wsdm_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'nlpcc-dbqa_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'xnli_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'input_ids': tf.FixedLenFeature(shape=[256], dtype=tf.int64, default_value=None), 
 # 'ant_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'chnsenticorp_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'ant_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'wsdm_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'ccks_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # # 'jd_comment_label_ids': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
 # 'chnsenticorp_loss_multiplier': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
 "input_mask":tf.FixedLenFeature([256], tf.int64),
 # 'masked_lm_positions':tf.FixedLenFeature([3], tf.int64)
 "task_id":tf.FixedLenFeature([], tf.int64),
 'masked_lm_positions':tf.FixedLenFeature([3], tf.int64),
 'masked_lm_ids':tf.FixedLenFeature([3], tf.int64),
 'masked_lm_weights':tf.FixedLenFeature([3], tf.int64),
 "input_ids":tf.FixedLenFeature([256], tf.int64),
}

	params = Bunch({})
	params.epoch = 1
	params.batch_size = 32

	def _decode_record(record, name_to_features):
		example = tf.parse_single_example(record, name_to_features)
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

	task_type_dict = ["ccks", "wsdm", "xnli", "nlpcc-dbqa",
					"ant", "lcqmc", "chnsenticorp"]

	def task_statics(record, task_type_dict, name_to_features):
		example = record #tf.parse_single_example(record, name_to_features)
		output_tensor = []
		for task_id, task in enumerate(task_type_dict):
			print(example["{}_mask".format(task)].get_shape())
			output_tensor.append(tf.reduce_sum(example["{}_mask".format(task)]))
		task_id = tf.argmax(tf.convert_to_tensor(output_tensor))
		return tf.cast(task_id, tf.int32)

	def train_batch_input_fn(input_file, _parse_fn, name_to_features,
		params, **kargs):
		if_shard = kargs.get("if_shard", "1")

		worker_count = kargs.get("worker_count", 1)
		task_index = kargs.get("task_index", 0)

		# dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))

		# def dataset_repeat(input_file):
		# 	output = []
		# 	for l in input_file:
		# 		dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
		# 		dataset = dataset.repeat()
		# 	output.append(dataset)
		# 	return output

		# output = dataset_repeat(input_file)

		dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_file))
		# dataset = dataset.shuffle(buffer_size=len(input_file))
		dataset = dataset.repeat(1)

		# `cycle_length` is the number of parallel files that get read.
		cycle_length = 1

		# `sloppy` mode means that the interleaving is not exact. This adds
		# even more randomness to the training pipeline.
		dataset = dataset.apply(
					tf.contrib.data.parallel_interleave(
					  tf.data.TFRecordDataset,
					  sloppy=True,
					  cycle_length=cycle_length,
					  block_length=16))

		print("==worker_count {}, task_index {}==".format(worker_count, task_index))
		if if_shard == "1":
			dataset = dataset.shard(worker_count, task_index)
		# dataset = dataset.shuffle(
		# 						buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
		# 						seed=np.random.randint(0,1e10,1)[0],
		# 						reshuffle_each_iteration=True)
		
		dataset = dataset.map(lambda x:_parse_fn(x, name_to_features),
							num_parallel_calls=kargs.get("num_parallel_calls", 10))
		# dataset = dataset.apply(tf.contrib.data.rejection_resample(
		# 		lambda x:task_statics(x, task_type_dict, name_to_features),
		# 		target_dist=1./tf.constant(np.ones(len(task_type_dict)).astype(np.float32))
		# 	))
		dataset = dataset.batch(params.get("batch_size", 32))
		iterator = dataset.make_one_shot_iterator()
		features = iterator.get_next()
		return features

	# input_file = ["/data/xuht/multi_task/data/merged_train_tfrecords"]

	input_file = ["/data/xuht/lcqmc_lm_train"
					# "/data/xuht/multi_task/data/single_sentence/chnsenticorp/train_tfrecords",
					# "/data/xuht/multi_task/data/sentence_pair/xnli/train_tfrecords",
					# "/data/xuht/multi_task/data/sentence_pair/lcqmc/train_tfrecords",
					# "/data/xuht/multi_task/data/sentence_pair/ccks/train_tfrecords",
					# "/data/xuht/multi_task/data/sentence_pair/wsdm/train_tfrecords",
					# "/data/xuht/multi_task/data/sentence_pair/ant/train_tfrecords",
					# "/data/xuht/multi_task/data/qa/nlpcc-dbqa/train_tfrecords"
					]
	# datasets = [train_batch_single_input_fn(file, _decode_record, name_to_features, params) for file in input_file]
	# # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
	# choice_dataset = tf.data.Dataset.range(len(input_file)).repeat()



	# file_test = "/data/xuht/porn/clean_data/textcnn/distillation/train_tfrecords"

	# input_fn = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

	input_fn = train_batch_input_fn(input_file,
						_decode_record, name_to_features, params)
	
	sess = tf.Session()
	
	init_op = tf.group(
				tf.local_variables_initializer())
	sess.run(init_op)
	
	i = 0
	cnt = 0
	feature = {}
	task_cnt = {}
	cnt = 0

	while True:
		try:
			features = sess.run(input_fn)
			tmp = {}
			for key in features:
				if "loss_multiplier" in key:
					tmp[key] = np.sum(features[key])
			for key in tmp:
				if tmp[key] > 0:
					if key in task_cnt:
						task_cnt[key] += tmp[key]
					else:
						task_cnt[key] = tmp[key]
			print(features['input_ids'].shape)
			
		except tf.errors.OutOfRangeError:
			print("End of dataset")
			break
	print(task_cnt)
	
	# while True:
	# 	try:
	# 		features = sess.run(input_fn)
	# 		for key in features:
	# 			if key in feature:
	# 				feature[key].extend(features[key].tolist())
	# 			else:
	# 				feature[key] = features[key].tolist()
	# 	except tf.errors.OutOfRangeError:
	# 		print("End of dataset")
	# 		break

	# for key in feature:
	# 	print(key, len(feature[key]))
	
	# data_index = np.random.permutation(len(f["input_ids"])).tolist()
	# _writer = tf.python_io.TFRecordWriter("/data/xuht/multi_task/merged_train_tfrecords")

	# for idx in data_index:
	# 	features = {}
	# 	features["input_ids"] = tf_data_utils.create_int_feature(feature["input_ids"][idx])
	# 	features["input_mask"] = tf_data_utils.create_int_feature(feature["input_mask"][idx])
	# 	features["segment_ids"] = tf_data_utils.create_int_feature(feature["segment_ids"][idx])
	# 	for task in task_type_dict:
	# 		features["{}_mask".format(task)] = tf_data_utils.create_int_feature([feature["{}_mask".format(task)][idx]])
	# 		features["{}_label_ids".format(task)] = tf_data_utils.create_int_feature([feature["{}_label_ids".format(task)][idx]])
	# 	tf_example = tf.train.Example(features=tf.train.Features(feature=features))
	# 	_writer.write(tf_example.SerializeToString())
test()