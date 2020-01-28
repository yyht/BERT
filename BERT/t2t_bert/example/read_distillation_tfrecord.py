import tensorflow as tf
from bunch import Bunch
import numpy as np
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

name_to_features = {
				"label_id":tf.FixedLenFeature([], tf.int64),
				"feature":tf.FixedLenFeature([768], tf.float32),
				"prob":tf.FixedLenFeature([5], tf.float32)
		        }

def read_distilaltion(input_data):

	graph = tf.Graph()
	with graph.as_default():
		

		params = Bunch({})
		params.epoch = 1
		params.batch_size = 32
		print(params["batch_size"], "===batch size===")

		def _decode_record(record, name_to_features):
			example = tf.parse_example(record, name_to_features)
			return example

		def train_input_fn(input_file, _parse_fn, name_to_features,
			params, **kargs):

			dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
			dataset = dataset.batch(params.get("batch_size", 32))
			dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
			dataset = dataset.repeat(params.get("epoch", 100))
			iterator = dataset.make_one_shot_iterator()
			features = iterator.get_next()
			return features

		input_fn = train_input_fn(input_data, _decode_record, name_to_features, params)
		
		sess = tf.Session()
		
		init_op = tf.group(
					tf.local_variables_initializer())
		sess.run(init_op)
		
		i = 0
		cnt = 0
		feature_dict_lst = []
		feature, prob, label_id = [], [], []
		while True:
			try:
				features = sess.run(input_fn)
				i += 1
				cnt += 1
				feature.extend(features["feature"].tolist())
				prob.extend(features["prob"].tolist())
				label_id.extend(features["label_id"].tolist())
			except tf.errors.OutOfRangeError:
				print("End of dataset")
				break
		for index in range(len(label_id)):
			tmp = {
					"feature":feature[index], 
					"prob":prob[index],
					"label_id":label_id[index]
					}
			feature_dict_lst.append(tmp)
		return feature_dict_lst