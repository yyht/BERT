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

import numpy as np
import tensorflow as tf
from bunch import Bunch

flags = tf.flags

FLAGS = flags.FLAGS
from data_generator import tf_data_utils
from data_generator import tokenization


epoch = 1

flags.DEFINE_string(
	"buckets", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"batch_size", 32,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_length", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"max_predictions_per_seq", None,
	"Input TF example files (can be a glob or comma separated).")

'''
end-to-end label noise robust models
'''

def _decode_record(record, name_to_features):
	"""Decodes a record to a TensorFlow example.

	name_to_features = {
	            "input_ids":
	                    tf.FixedLenFeature([max_seq_length], tf.int64),
	            "input_mask":
	                    tf.FixedLenFeature([max_seq_length], tf.int64),
	            "segment_ids":
	                    tf.FixedLenFeature([max_seq_length], tf.int64),
	            "masked_lm_positions":
	                    tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
	            "masked_lm_ids":
	                    tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
	            "masked_lm_weights":
	                    tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
	            "next_sentence_labels":
	                    tf.FixedLenFeature([1], tf.int64),
	    }

	"""
	example = tf.parse_example(record, name_to_features)
	return example

def train_input_fn(input_file, _parse_fn, name_to_features,
	params, **kargs):
	if_shard = kargs.get("if_shard", "0")

	worker_count = kargs.get("worker_count", 1)
	task_index = kargs.get("task_index", 0)

	dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
	print("==worker_count {}, task_index {}==".format(worker_count, task_index))
	if if_shard == "1":
	    dataset = dataset.shard(worker_count, task_index)
	dataset = dataset.shuffle(
	                        buffer_size=params.get("buffer_size", 1024)+3*params.get("batch_size", 32),
	                        seed=np.random.randint(0,1e10,1)[0],
	                        reshuffle_each_iteration=True)
	dataset = dataset.batch(params.get("batch_size", 32))
	dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))


	dataset = dataset.repeat(params.get("epoch", 100))
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()
	return features

def main(_):

	graph = tf.Graph()
	with graph.as_default():
		sess_config = tf.ConfigProto()
		import random
		name_to_features = {
				"input_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"masked_lm_positions":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
					tf.FixedLenFeature([], tf.int64),
				}

		params = Bunch({})
		params.epoch = 1
		params.batch_size = FLAGS.batch_size
		def parse_folder(path):
			files = os.listdir(path)
			output = []
			for file_name in files:
				output.append(os.path.join(path, file_name))
			random.shuffle(output)
			return output
		print(params["batch_size"], "===batch size===")
		input_fn = train_input_fn(os.path.join(FLAGS.buckets, FLAGS.train_file), _decode_record, name_to_features, params)
		
		sess = tf.Session(config=sess_config)
		
		init_op = tf.group(
					tf.local_variables_initializer())
		sess.run(init_op)

		tokenizer = tokenization.FullTokenizer(
			vocab_file='./data/chinese_L-12_H-768_A-12/vocab.txt', 
			do_lower_case=True,
			do_whole_word_mask=True)
		
		i = 0
		cnt = 0
		next_sentence = []
		while True:
			try:
				features = sess.run(input_fn)
				masked_lm_ids = features['masked_lm_ids'].tolist()[0]
				masked_lm_positions = features['masked_lm_positions'].tolist()[0]
				input_ids = features['input_ids'].tolist()[0]

				input_token = tokenizer.convert_ids_to_tokens([ids for ids in input_ids if ids != 0])
				print('==before mlm==', "".join(input_token))
				print(masked_lm_ids, '=====mlm======')
				for i,j in zip(masked_lm_positions, masked_lm_ids):
					input_token[i] = tokenizer.inv_vocab[j]
				print('==after revocer mlm==', "".join(input_token))
				if cnt == 5:
					break
				i += 1
				cnt += features['next_sentence_labels'].shape[0]
			except tf.errors.OutOfRangeError:
				print("End of dataset")
				break
		print(cnt)
if __name__ == "__main__":
	tf.app.run()