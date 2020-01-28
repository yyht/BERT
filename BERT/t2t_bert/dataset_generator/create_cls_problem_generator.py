
import random
from copy import copy
import numpy as np

import tensorflow as tf
from data_generator import tf_data_utils
from dataset_generator.create_masked_lm_generator import create_masked_lm_predictions
from dataset_generator.dataset_utils import create_mask_and_padding
from data_generator import tokenization

def create_cls_problem_generator(task_type,
									examples,
									label_dict,
									multi_task_config,
									tokenizer,
									mode):
	max_seq_length = multi_task_config[task_type]["max_length"]
	lm_augumentation = multi_task_config[task_type]["lm_augumentation"]
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = None
		if example.text_b:
			try:
				tokens_b = tokenizer.tokenize(example.text_b)
			except:
				print("==token b error==", example.text_b, ex_index)
				break

		if tokens_b:
			tf_data_utils._truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]

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

		if lm_augumentation and mode == 'train':
			rng = random.Random()
			(mask_lm_tokens, masked_lm_positions,
				masked_lm_labels) = create_masked_lm_predictions(
					tokens,
					multi_task_config[task_type]["masked_lm_prob"],
					multi_task_config[task_type]["max_predictions_per_seq"],
					list(tokenizer.vocab.keys()), rng)

			_, mask_lm_tokens, _ = create_mask_and_padding(
				mask_lm_tokens, copy(segment_ids), max_seq_length)
			masked_lm_weights, masked_lm_labels, masked_lm_positions = create_mask_and_padding(
				masked_lm_labels, masked_lm_positions, 
				multi_task_config[task_type]["max_predictions_per_seq"])
			mask_lm_input_ids = tokenizer.convert_tokens_to_ids(
				mask_lm_tokens)
			masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

			assert len(mask_lm_tokens) == max_seq_length

		input_mask, tokens, segment_ids = create_mask_and_padding(
			tokens, segment_ids, max_seq_length)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if ex_index < 5:
			tf.logging.debug("*** Example ***")
			tf.logging.debug("tokens: %s" % " ".join(
				[tokenization.printable_text(x) for x in tokens]))
			tf.logging.debug("input_ids: %s" %
							 " ".join([str(x) for x in input_ids]))
			tf.logging.debug("input_mask: %s" %
							 " ".join([str(x) for x in input_mask]))
			tf.logging.debug("segment_ids: %s" %
							 " ".join([str(x) for x in segment_ids]))
			tf.logging.debug("%s_label_ids: %s" %
							 (task_type, str(label_id)))
			tf.logging.debug("%s_label: %s" %
							 (task_type, str(example.label)))
			if lm_augumentation and mode == 'train':
				tf.logging.debug("mask lm tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in mask_lm_tokens]))
				tf.logging.debug("mask lm input_ids: %s" %
								 " ".join([str(x) for x in mask_lm_input_ids]))
				tf.logging.debug("mask lm label ids: %s" %
								 " ".join([str(x) for x in masked_lm_ids]))
				tf.logging.debug("mask lm position: %s" %
								 " ".join([str(x) for x in masked_lm_positions]))
			
		if not lm_augumentation:
			return_dict = {
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids,
				'%s_label_ids' % task_type: label_id
			}

		else:
			if mode == 'train':
				return_dict = {
					'input_ids': mask_lm_input_ids,
					'input_mask': input_mask,
					'segment_ids': segment_ids,
					'%s_label_ids' % task_type: label_id,
					"masked_lm_positions": masked_lm_positions,
					"masked_lm_ids": masked_lm_ids,
					"masked_lm_weights": masked_lm_weights,
				}
			else:
				return_dict = {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    '%s_label_ids' % task_type: label_id,
                    "masked_lm_positions": [0]*multi_task_config[task_type]["max_predictions_per_seq"],
                    "masked_lm_ids": [0]*multi_task_config[task_type]["max_predictions_per_seq"],
                    "masked_lm_weights": [0]*multi_task_config[task_type]["max_predictions_per_seq"],
                }
		
		yield return_dict





	

	