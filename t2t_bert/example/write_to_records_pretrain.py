import tensorflow as tf
from data_generator import tf_data_utils
from data_generator import tokenization
import collections
from example.feature_writer import PairPreTrainingFeature
from data_generator import pretrain_feature
import multiprocessing
import random
import copy
import numpy as np

def per_seq_dupe_func(tokens_a, tokens_b, **kargs):
	masked_lm_prob = kargs["masked_lm_prob"]
	max_num_tokens = kargs["max_num_tokens"]

	max_predictions_per_seq = kargs["max_predictions_per_seq"]
	dupe_factor_actual = kargs["dupe_factor"]

	if tokens_b:
		tokens_b_ = tokens_b
	else:
		tokens_b_ = []

	total_len = len(tokens_a) + len(tokens_b_)

	max_predictions_per_seq_actual = int(total_len/25)
	if max_predictions_per_seq_actual == 0:
		max_predictions_per_seq_actual = 1

	max_predictions_per_seq_actual = min([max_predictions_per_seq, max_predictions_per_seq_actual])
	
	dupe_factor_actual = min([dupe_factor_actual, 2*max_predictions_per_seq_actual])

	return max_predictions_per_seq_actual, dupe_factor_actual

def create_instances_qa(examples, dupe_factor, max_seq_length, 
					masked_lm_prob, tokenizer, 
					max_predictions_per_seq,
					rng,
					per_seq_dupe_func):
	vocab_words = list(tokenizer.vocab.keys())
	instances = []
	for example in examples:
		max_num_tokens = max_seq_length - 3
		tokens_a_ = tokenizer.tokenize(example.text_a)
		tokens_b_ = tokenizer.tokenize(example.text_b) 

		[max_predictions_per_seq_actual,
		dupe_factor_actual] = per_seq_dupe_func(tokens_a_, tokens_b_,
											masked_lm_prob=masked_lm_prob,
											max_num_tokens=max_num_tokens,
											max_predictions_per_seq=max_predictions_per_seq,
											dupe_factor=dupe_factor)

		tf.logging.info("max_predictions_per_seq:{} dupe_factor_actual:{}".format(max_predictions_per_seq_actual,
			dupe_factor_actual))

		for _ in range(dupe_factor_actual):

			tokens_a = copy.deepcopy(tokens_a_)
			tokens_b = copy.deepcopy(tokens_b_)

			tf_data_utils._truncate_seq_pair_v1(tokens_a, tokens_b, max_num_tokens, rng)

			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in tokens_a:
				tokens.append(token)
				segment_ids.append(0)

			tokens.append("[SEP]")
			segment_ids.append(0)

			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			(tokens, masked_lm_positions,
			 masked_lm_labels) = tf_data_utils.create_masked_lm_predictions(
				 tokens, masked_lm_prob, max_predictions_per_seq_actual, vocab_words, rng)
			instance = pretrain_feature.PreTrainingInstance(
				guid=example.guid,
				tokens=tokens,
				segment_ids=segment_ids,
				is_random_next=0,
				label=example.label,
				masked_lm_positions=masked_lm_positions,
				masked_lm_labels=masked_lm_labels)
			instances.append(instance)

	return instances

def create_instances_classification(examples, dupe_factor, max_seq_length, 
					masked_lm_prob, tokenizer, 
					max_predictions_per_seq,
					rng,
					per_seq_dupe_func):
	vocab_words = list(tokenizer.vocab.keys())
	instances = []
	for example in examples:
		tokens_a_ = tokenizer.tokenize(example.text_a)
		
		tokens_b_ = None
		if example.text_b:
			max_num_tokens = max_seq_length - 3
			try:
				tokens_b_ = tokenizer.tokenize(example.text_b)
			except:
				print("==token b error==", example.text_b, ex_index)
				break
		else:
			max_num_tokens = max_seq_length - 2

		[max_predictions_per_seq_actual,
		dupe_factor_actual] = per_seq_dupe_func(tokens_a_, None, masked_lm_prob=masked_lm_prob,
											max_num_tokens=max_num_tokens,
											max_predictions_per_seq=max_predictions_per_seq,
											dupe_factor=dupe_factor)

		tf.logging.info("max_predictions_per_seq:{} dupe_factor_actual:{}".format(max_predictions_per_seq_actual,
			dupe_factor_actual))

		for _ in range(dupe_factor_actual):

			tokens_a = copy.deepcopy(tokens_a_)
			if tokens_b_:
				tokens_b = copy.deepcopy(tokens_b_)
			else:
				tokens_b = None

			if tokens_b:
				tf_data_utils._truncate_seq_pair_v1(tokens_a, tokens_b, max_num_tokens, rng)
			else:
				tf_data_utils._truncate_seq(tokens_a, max_num_tokens, rng)

			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)

			for token in tokens_a:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			if tokens_b:
				for token in tokens_b:
					tokens.append(token)
					segment_ids.append(1)
				tokens.append("[SEP]")
				segment_ids.append(1)

			(tokens, masked_lm_positions,
			 masked_lm_labels) = tf_data_utils.create_masked_lm_predictions(
				 tokens, masked_lm_prob, max_predictions_per_seq_actual, vocab_words, rng)
			instance = pretrain_feature.PreTrainingInstance(
				guid=example.guid,
				tokens=tokens,
				segment_ids=segment_ids,
				is_random_next=0,
				label=example.label,
				masked_lm_positions=masked_lm_positions,
				masked_lm_labels=masked_lm_labels)
			instances.append(instance)

	return instances
	
def build_chunk(examples, chunk_num=10):
	"""
	split list into sub lists:分块
	:param lines: total thing
	:param chunk_num: num of chunks
	:return: return chunks but the last chunk may not be equal to chunk_size
	"""
	total = len(examples)
	chunk_size = float(total) / float(chunk_num + 1)
	chunks = []
	for i in range(chunk_num + 1):
		if i == chunk_num:
			chunks.append(examples[int(i * chunk_size):])
		else:
			chunks.append(examples[int(i * chunk_size):int((i + 1) * chunk_size)])
	return chunks

def multi_process(examples, process_num, 
				label_dict,
				tokenizer, 
				max_seq_length,
				masked_lm_prob, 
				max_predictions_per_seq, 
				output_file,
				dupe,
				random_seed=2018,
				feature_type="pretrain_qa",
				log_cycle=100,
				per_seq_dupe_func=per_seq_dupe_func):

	chunk_num = process_num - 1

	chunks = build_chunk(examples, chunk_num)
	pool = multiprocessing.Pool(processes=process_num)

	for chunk_id, each_chunk in enumerate(chunks):
		output_file_ = output_file + "/chunk_{}.tfrecords".format(chunk_id)
		print("#mask_language_model_multi_processing.length of chunk:",len(each_chunk),";file_name:",output_file_,";chunk_id:",chunk_id)
		pool.apply_async(write_instance_to_example_files,
			args=(each_chunk, label_dict,tokenizer,max_seq_length,
					masked_lm_prob,max_predictions_per_seq,
					output_file_, dupe,random_seed,feature_type,log_cycle,
					per_seq_dupe_func)) # apply_async
	pool.close()
	pool.join()

def write_instance_to_example_files(examples, 
									label_dict,
									tokenizer, 
									max_seq_length,
									masked_lm_prob, 
									max_predictions_per_seq, 
									output_file,
									dupe,
									random_seed=2018,
									feature_type="pretrain_qa",
									log_cycle=100,
									per_seq_dupe_func=per_seq_dupe_func):


	"""Create TF example files from `TrainingInstance`s."""
	rng = random.Random(random_seed)

	feature_writer = PairPreTrainingFeature(output_file, is_training=False)

	if feature_type == "pretrain_qa":
		instances = create_instances_qa(examples, dupe, 
								max_seq_length,
								masked_lm_prob, 
								tokenizer,
								max_predictions_per_seq,
								rng,
								per_seq_dupe_func)
	elif feature_type == "pretrain_classification":
		instances = create_instances_classification(examples, dupe, 
								max_seq_length,
								masked_lm_prob, 
								tokenizer,
								max_predictions_per_seq,
								rng,
								per_seq_dupe_func)

	rng.shuffle(instances)

	total_written = 0
	for (inst_index, instance) in enumerate(instances):
		input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
		input_mask = [1] * len(input_ids)
		segment_ids = list(instance.segment_ids)
		assert len(input_ids) <= max_seq_length

		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		masked_lm_positions = list(instance.masked_lm_positions)
		masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
		masked_lm_weights = [1.0] * len(masked_lm_ids)

		while len(masked_lm_positions) < max_predictions_per_seq:
			masked_lm_positions.append(0)
			masked_lm_ids.append(0)
			masked_lm_weights.append(0.0)

		if len(instance.label) == 1:
			label_id = label_dict[instance.label[0]]

		features = pretrain_feature.PreTrainingFeature(
					guid=instance.guid, 
					input_ids=input_ids,
					input_mask=input_mask, 
					segment_ids=segment_ids,
					masked_lm_positions=masked_lm_positions, 
					masked_lm_ids=masked_lm_ids,
					masked_lm_weights=masked_lm_weights,
					label_ids=label_id,
					is_random_next=instance.is_random_next)
		feature_writer.process_feature(features)

		if np.mod(inst_index, log_cycle) == 0:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (features.guid))
			tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in instance.tokens]))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			tf.logging.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			tf.logging.info(
					"masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions]))
			tf.logging.info(
					"masked_lm_ids: %s" % " ".join([str(x) for x in masked_lm_ids]))
			tf.logging.info(
					"masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights]))
			tf.logging.info("label: {} (id = {})".format(instance.label, label_id))
			tf.logging.info("is_random_next: {} ".format(instance.is_random_next))
			tf.logging.info("length of tokens: {} ".format(len(instance.tokens)))
		
	feature_writer.close()

	tf.logging.info("Wrote %d total instances", total_written) 


