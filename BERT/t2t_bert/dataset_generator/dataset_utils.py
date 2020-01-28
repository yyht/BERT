
import os
import unicodedata
import random
import collections
# -*- coding: utf-8 -*-
import numpy as np

def punc_augument(raw_inputs, params):
	for char_ind, char in enumerate(raw_inputs):
		if char in params.punc_list:
			if random.uniform(0, 1) <= params.punc_replace_prob:
				raw_inputs[char_ind] = random.choice(params.punc_list)

	return raw_inputs

def get_dirty_text_ind(text):
	"""Performs invalid character removal and whitespace cleanup on text."""

	text = [unicodedata.normalize("NFD", t) for t in text]
	output = []
	for char_ind, char in enumerate(text):
		if len(char) > 1:
			output.append(char_ind)
			continue
		cp = ord(char)
		if cp == 0 or cp == 0xfffd or _is_control(char):
			output.append(char_ind)

	return output

def create_mask_and_padding(tokens, segment_ids, max_length):

	input_mask = [1]*len(tokens)
	pad_list = ['[PAD]'] * (max_length - len(input_mask))

	input_mask += [0]*len(pad_list)
	segment_ids += [0]*len(pad_list)
	tokens += pad_list

	return input_mask, tokens, segment_ids

# get dummy labels
def _create_dummpy_label(task_type, max_length):
    if task_type == 'cls_task':
        return 0
    else:
        return [0]*max_length