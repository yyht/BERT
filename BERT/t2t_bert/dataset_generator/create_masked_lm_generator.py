
import tensorflow as tf
import numpy as np

import os
import unicodedata
import random
import collections
import random
from copy import copy
import numpy as np

import tensorflow as tf
from data_generator import tf_data_utils
from dataset_generator.dataset_utils import create_mask_and_padding
from data_generator import tokenization

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
										  ["index", "label"])

TrainingInstance = collections.namedtuple("TrainingInstance",
										  ['tokens', 'segment_ids',
										   'masked_lm_positions',
										   'masked_lm_labels',
										   'is_random_next'])									  

def create_masked_lm_predictions(tokens, masked_lm_prob,
								 max_predictions_per_seq, vocab_words, rng):
	"""Creates the predictions for the masked LM objective."""

	cand_indexes = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]":
			continue
		cand_indexes.append(i)

	rng.shuffle(cand_indexes)

	output_tokens = list(tokens)

	num_to_predict = min(max_predictions_per_seq,
						 max(1, int(round(len(tokens) * masked_lm_prob))))

	masked_lms = []
	covered_indexes = set()
	for index in cand_indexes:
		if len(masked_lms) >= num_to_predict:
			break
		if index in covered_indexes:
			continue
		covered_indexes.add(index)

		masked_token = None
		# 80% of the time, replace with [MASK]
		if rng.random() < 0.8:
			masked_token = "[MASK]"
		else:
			# 10% of the time, keep original
			if rng.random() < 0.5:
				masked_token = tokens[index]
			# 10% of the time, replace with random word
			else:
				masked_token = vocab_words[rng.randint(
					0, len(vocab_words) - 1)]

		output_tokens[index] = masked_token

		masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

	masked_lms = sorted(masked_lms, key=lambda x: x.index)

	masked_lm_positions = []
	masked_lm_labels = []
	for p in masked_lms:
		masked_lm_positions.append(p.index)
		masked_lm_labels.append(p.label)

	return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_instances_from_document(
		all_documents, document_index, max_seq_length, short_seq_prob,
		masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
	"""Creates `TrainingInstance`s for a single document."""
	document = all_documents[document_index]

	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 3

	# We *usually* want to fill up the entire sequence since we are padding
	# to `max_seq_length` anyways, so short sequences are generally wasted
	# computation. However, we *sometimes*
	# (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
	# sequences to minimize the mismatch between pre-training and fine-tuning.
	# The `target_seq_length` is just a rough target however, whereas
	# `max_seq_length` is a hard limit.
	target_seq_length = max_num_tokens
	if rng.random() < short_seq_prob:
		target_seq_length = rng.randint(2, max_num_tokens)

	# We DON'T just concatenate all of the tokens from a document into a long
	# sequence and choose an arbitrary split point because this would make the
	# next sentence prediction task too easy. Instead, we split the input into
	# segments "A" and "B" based on the actual "sentences" provided by the user
	# input.
	instances = []
	current_chunk = []
	current_length = 0
	i = 0
	while i < len(document):
		segment = document[i]
		current_chunk.append(segment)
		current_length += len(segment)
		if i == len(document) - 1 or current_length >= target_seq_length:
			if current_chunk:
				# `a_end` is how many segments from `current_chunk` go into the `A`
				# (first) sentence.
				a_end = 1
				if len(current_chunk) >= 2:
					a_end = rng.randint(1, len(current_chunk) - 1)

				tokens_a = []
				for j in range(a_end):
					tokens_a.extend(current_chunk[j])

				tokens_b = []
				# Random next
				is_random_next = False
				if len(current_chunk) == 1 or rng.random() < 0.5:
					is_random_next = True
					target_b_length = target_seq_length - len(tokens_a)

					# This should rarely go for more than one iteration for large
					# corpora. However, just to be careful, we try to make sure that
					# the random document is not the same as the document
					# we're processing.
					for _ in range(10):
						random_document_index = rng.randint(
							0, len(all_documents) - 1)
						if random_document_index != document_index:
							break

					random_document = all_documents[random_document_index]
					random_start = rng.randint(0, len(random_document) - 1)
					for j in range(random_start, len(random_document)):
						tokens_b.extend(random_document[j])
						if len(tokens_b) >= target_b_length:
							break
					# We didn't actually use these segments so we "put them back" so
					# they don't go to waste.
					num_unused_segments = len(current_chunk) - a_end
					i -= num_unused_segments
				# Actual next
				else:
					is_random_next = False
					for j in range(a_end, len(current_chunk)):
						tokens_b.extend(current_chunk[j])
				tf_data_utils._truncate_seq_pair_v1(tokens_a, tokens_b,
								  max_num_tokens, rng)
				if len(tokens_a) < 1 or len(tokens_b) < 1:
					current_chunk = []
					current_length = 0
					i += 1
					continue
				assert len(tokens_a) >= 1, tokens_a
				assert len(tokens_b) >= 1, tokens_b

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
				 masked_lm_labels) = create_masked_lm_predictions(
					 tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
				instance = TrainingInstance(
					tokens=tokens,
					segment_ids=segment_ids,
					is_random_next=is_random_next,
					masked_lm_positions=masked_lm_positions,
					masked_lm_labels=masked_lm_labels)
				instances.append(instance)
			current_chunk = []
			current_length = 0
		i += 1

	return instances