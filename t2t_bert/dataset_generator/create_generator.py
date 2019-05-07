from data_reader import SentenceProcessor, SentencePairProcessor
from data_generator import tokenization
from create_cls_problem_generator import create_cls_problem_generator
from create_masked_lm_generator import create_instances_from_document
from create_pretrain_generator import create_pretraining_generator
from dataset_generator.dataset_utils import _create_dummpy_label
from dataset_generator.problem_generator import problem_generator
import random
import tensorflow as tf
import numpy as np
import os

global sample_cnt

def read_data_fn(FLAGS, multi_task_config, task, mode):

	train_file = os.path.join(FLAGS.buckets, multi_task_config[task]["train_file"])

	test_file = os.path.join(FLAGS.buckets, multi_task_config[task]["test_file"])

	dev_file = os.path.join(FLAGS.buckets, multi_task_config[task]["dev_file"])

	label_id = os.path.join(FLAGS.buckets, multi_task_config[task]["label_id"])

	print(train_file, test_file, dev_file, label_id, task, "======")

	tokenizer = tokenization.FullTokenizer(
                    vocab_file=multi_task_config[task]["vocab_file"], 
                    do_lower_case=multi_task_config[task]["do_lower_case"])

	data_type = multi_task_config[task]["data_type"]
	if data_type == "single_sentence":
		classifier_data_api = SentenceProcessor()
	elif data_type == "sentence_pair":
		classifier_data_api = SentencePairProcessor()
	classifier_data_api.get_labels(label_id)

	if mode == "train":
		examples = classifier_data_api.get_train_examples(train_file,
											is_shuffle=True) 
	elif mode == "eval":
		examples = classifier_data_api.get_train_examples(dev_file,
											is_shuffle=False)
	elif mode == "test":
		examples = classifier_data_api.get_train_examples(test_file,
											is_shuffle=False)
	else:
		examples = None

	return {
		"examples":examples,
		"tokenizer":tokenizer,
		"label2id":classifier_data_api.label2id
	}

def get_generator(examples, multi_task_config, task_type, mode, label2id, tokenizer):
	example_generator = problem_generator(task_type, examples, label2id, 
									multi_task_config, tokenizer, mode)
	data_num = len(examples)
	return {
		"generator":example_generator,
		"data_num":data_num
	}

def create_generator(FLAGS, multi_task_config, mode, epoch):
	"""Function to create iterator for multiple problem
	This function dose the following things:
	1. Create dummy labels for each problems.
	2. Initialize all generators
	3. Sample a problem to train at this batch. If eval, take turns
	4. Create a loss multiplier
	5. Tried to generate samples for target problem, if failed, init gen
	6. Add dummy label to other problems
	Example:
		Problem: CWS|NER|weibo_ner&weibo_cws
		1. Dummy labels: CWS_label_ids: [0,0,0] ...
		2. Blablabla
		3. Sample, say (weibo_ner&weibo_cws)
		4. loss multipliers: {'CWS_loss_multiplier': 0, ..., 'weibo_ner_loss_multiplier': 1, ...}
		...
	Arguments:
		params {Params} -- params
		mode {mode} -- mode
		epoch {int} -- epochs to run
	"""
	# example
	# problem_list: ['NER', 'CWS', 'weibo_ner', 'weibo_cws']
	# problem_chunk: [['NER'], ['CWS'], ['weibo_ner', 'weibo_cws']]
	problem_list = []
	problem_chunk = []
	problem_id_dict = {}
	for idx, task_type in enumerate(FLAGS.multi_task_type.split(",")):
		problem_list += [task_type]
		problem_chunk.append([task_type])
		problem_id_dict[task_type] = idx

	print(problem_list, problem_chunk)

	# get dummy labels
	dummy_label_dict = {}
	for problem in problem_list:
		if multi_task_config[problem]["task_type"] != "pretrain":
			dummy_label_dict[problem+"_label_ids"] = _create_dummpy_label(
													multi_task_config[problem]["task_type"], 
													multi_task_config[problem]["max_length"])

	# init gen
	data_dict = {}
	for problem in problem_list:
		output = read_data_fn(FLAGS, multi_task_config, problem, mode)
		data_dict[problem] = {
			"examples":output["examples"],
			"tokenizer":output["tokenizer"],
			"label2id":output["label2id"]
		}

	gen_dict = {}
	for problem in problem_list:
		gen_dict[problem] = get_generator(data_dict[problem]["examples"],
			 							multi_task_config, 
			 							problem, 
			 							mode, 
			 							data_dict[problem]["label2id"],
			 							data_dict[problem]["tokenizer"])

	data_num_list = [gen_dict[chunk[0]]["data_num"]
							 for chunk in problem_chunk]

	sample_prob = np.array(data_num_list) / np.sum(data_num_list)
	print("==data_balanced==", sample_prob)
	sample_prob = np.array(
					[1]*len(data_num_list)) / np.sum([1]*len(data_num_list))
	print("==problem_balanced==", sample_prob)

	for index, key in enumerate(problem_chunk):
		print(key, data_num_list[index], "==data samples information==")

	data_iterator_dict = {}
	for problem in problem_list:
		data_iterator_dict[problem] = 0

	sample_cnt = 0
	previous_problem_chunk = []
	while gen_dict:
		# sample problem to train
		print(data_iterator_dict, "==problem iterator==")
		if len(problem_chunk) > 1:
			data_num_list = [gen_dict[chunk[0]]["data_num"]
							 for chunk in problem_chunk]
			if np.mod(sample_cnt, FLAGS.batch_size) == 0:
				if FLAGS.multitask_balance_type == 'data_balanced':
					sample_prob = np.array(data_num_list) / np.sum(data_num_list)
					current_problem_chunk_ind = np.random.choice(
						list(range(len(problem_chunk))), p=sample_prob)
					current_problem_chunk = problem_chunk[current_problem_chunk_ind]
				elif FLAGS.multitask_balance_type == 'problem_balanced':
					sample_prob = np.array(
						[1]*len(data_num_list)) / np.sum([1]*len(data_num_list))
					current_problem_chunk_ind = np.random.choice(
						list(range(len(problem_chunk))), p=sample_prob)
					current_problem_chunk = problem_chunk[current_problem_chunk_ind]
				previous_problem_chunk = current_problem_chunk
				print(sample_cnt, current_problem_chunk)
			else:
				current_problem_chunk = previous_problem_chunk
		else:
			current_problem_chunk = problem_chunk[0]
		sample_cnt += 1

		# create loss multiplier
		loss_multiplier = {}
		for problem in problem_list:
			if problem in current_problem_chunk:
				loss_multiplier[problem+'_loss_multiplier'] = 1
			else:
				loss_multiplier[problem+'_loss_multiplier'] = 0

		base_dict = {}
		base_input = None

		for problem in current_problem_chunk:
			try:
				instance = next(gen_dict[problem]["generator"])
			except StopIteration:
				if mode == 'train':
					gen_dict[problem] = get_generator(data_dict[problem]["examples"],
							 							multi_task_config, 
							 							problem, 
							 							mode, 
							 							data_dict[problem]["label2id"],
							 							data_dict[problem]["tokenizer"])
					tf.logging.info("re-create generator")
					instance = next(gen_dict[problem]["generator"])
					data_iterator_dict[problem] += 1
				else:
					del gen_dict[problem]
					continue
			except KeyError:
				continue

			base_dict.update(instance)
			if base_input is None:
				base_input = instance['input_ids']
			base_dict.update(
					{"task_id":problem_id_dict[problem]}
				)

		itera_num = [data_iterator_dict[problem] for problem in data_iterator_dict]
		cnt = 0
		for problem in data_iterator_dict:
			if data_iterator_dict[problem] >= epoch:
				cnt += 1
		if cnt == len(data_iterator_dict):
			break

		if not base_dict:
			continue

		# add dummpy labels
		for dummy_problem in dummy_label_dict:
			if dummy_problem not in base_dict:
				base_dict[dummy_problem] = dummy_label_dict[dummy_problem]

		# add loss multipliers
		base_dict.update(loss_multiplier)
		yield base_dict