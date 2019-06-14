from exmaple import classifier_prosessor
import numpy as np
import tensorflow as tf
from bunch import Bunch
from data_generator import tokenization
from data_generator import tf_data_utils
import json

def train_input_fn(FLAGS):

	multi_task_config = Bunch(json.load(open(os.path.join(FLAGS.buckets, FLAGS.multitask_dict))))

	vocab_path = FLAGS.vocab_file

	train_file_dict = {}
	test_file_dict = {}
	dev_file_dict = {}
	train_result_dict = {}
	test_result_dict = {}
	dev_result_dict = {}
	label_id_dict = {}
	for task in multi_task_config:
		train_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["train_file"])

		test_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["test_file"])

		dev_file_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["dev_file"])

		label_id_dict[task] = os.path.join(FLAGS.buckets, 
											multi_task_config[task]["label_id"])

	tokenizer = tokenization.FullTokenizer(
			vocab_file=vocab_path, 
			do_lower_case=FLAGS.lower_case)

	index = 0
	task_type_id = OrderedDict()
	label2id_dict = {}

	for task in (FLAGS.multi_task_type.split(",")):
		if task not in multi_task_config:
			continue
		task_type_id[task] = multi_task_config[task]
		index += 1
		data_type = multi_task_config[task]["data_type"]
		if data_type == "single_sentence":
			classifier_data_api = classifier_processor.SentenceProcessor()
			classifier_data_api.get_labels(label_id_dict[task])
		elif data_type == "sentence_pair":
			classifier_data_api = classifier_processor.SentencePairProcessor()
			classifier_data_api.get_labels(label_id_dict[task])

		train_examples = classifier_data_api.get_train_examples(train_file_dict[task],
											is_shuffle=True)
		label2id_dict[task] = classifier_data_api.label2id
		
		for item in train_examples:
			tmp = {"example":item,"task":task}
			total_examples.append(tmp)

	

	print(task_type_id.keys())
	print("==total data==", len(total_examples))

