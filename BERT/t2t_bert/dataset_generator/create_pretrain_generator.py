import random
from copy import copy
import numpy as np

import tensorflow as tf
from data_generator import tf_data_utils
from dataset_generator.create_masked_lm_generator import create_instances_from_document
from dataset_generator.dataset_utils import create_mask_and_padding
from data_generator import tokenization

def create_pretraining_generator(task_type,
								 inputs_list,
								 tokenizer,
								 multi_task_config
								 ):
	"""Slight modification of original code
	Raises:
		ValueError -- Input format not right
	"""

	all_documents = []
	for document in inputs_list:
		all_documents.append([])
		for sentence in document:
			all_documents[-1].append(tokenizer.tokenize('\t'.join(sentence)))

	all_documents = [d for d in all_documents if d]
	rng = random.Random()
	rng.shuffle(all_documents)

	vocab_words = list(tokenizer.vocab.keys())
	instances = []

	print_count = 0

	max_seq_length = multi_task_config[task_type]["max_length"]

	for _ in range(multi_task_config[task_type]["dupe_factor"]):
		for document_index in range(len(all_documents)):
			instances = create_instances_from_document(
				all_documents,
				document_index,
				max_seq_length,
				multi_task_config[task_type]["short_seq_prob"],
				multi_task_config[task_type]["masked_lm_prob"],
				multi_task_config[task_type]["max_predictions_per_seq"],
				vocab_words, rng)
			for instance in instances:
				tokens = instance.tokens
				segment_ids = list(instance.segment_ids)

				input_mask, tokens, segment_ids = create_mask_and_padding(
					tokens, segment_ids, max_seq_length)
				masked_lm_positions = list(instance.masked_lm_positions)
				masked_lm_weights, masked_lm_labels, masked_lm_positions = create_mask_and_padding(
					instance.masked_lm_labels, masked_lm_positions, 
					multi_task_config[task_type]["max_predictions_per_seq"])
				input_ids = tokenizer.convert_tokens_to_ids(tokens)
				masked_lm_ids = tokenizer.convert_tokens_to_ids(
					masked_lm_labels)
				next_sentence_label = 1 if instance.is_random_next else 0

				yield_dict = {
					"input_ids": input_ids,
					"input_mask": input_mask,
					"segment_ids": segment_ids,
					"masked_lm_positions": masked_lm_positions,
					"masked_lm_ids": masked_lm_ids,
					"masked_lm_weights": masked_lm_weights,
					"next_sentence_label_ids": next_sentence_label
				}

				if print_count < 3:
					tf.logging.debug('%s : %s' %
									 ('tokens', ' '.join([str(x) for x in tokens])))
					for k, v in yield_dict.items():
						if not isinstance(v, int):
							tf.logging.debug('%s : %s' %
											 (k, ' '.join([str(x) for x in v])))
					print_count += 1

				yield yield_dict