!pip install ../kernel/my_jieba.zip
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import re

"""Tokenization classes."""

import collections
import unicodedata
import six
import tensorflow as tf

def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
	"""Returns text encoded in a way suitable for print or `tf.logging`."""

	# These functions want `str` for both Python2 and Python3, but in one case
	# it's a Unicode string and in the other it's a byte string.
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text
		elif isinstance(text, unicode):
			return text.encode("utf-8")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
	"""Loads a vocabulary file into a dictionary."""
	vocab = collections.OrderedDict()
	index = 0
	with tf.gfile.GFile(vocab_file, "r") as reader:
		while True:
			token = convert_to_unicode(reader.readline())
			if not token:
				break
			token = token.strip()
			vocab[token] = index
			index += 1
	return vocab


def convert_by_vocab(vocab, items):
	"""Converts a sequence of [tokens|ids] using the vocab."""
	output = []
	for item in items:
		if item.startswith("##") and item.split("##")[-1] in vocab:
			if len(item.split("##")[-1]) == 1:
				cp = ord(item.split("##")[-1])
				if _is_chinese_char(cp):
					output.append(vocab.get(item.split("##")[-1], vocab["[UNK]"]))
				else:
					output.append(vocab.get(item, vocab["[UNK]"]))
			else:
				output.append(vocab.get(item, vocab["[UNK]"]))
		else:
			output.append(vocab.get(item, vocab["[UNK]"]))
	return output


def convert_tokens_to_ids(vocab, tokens):
	return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
	def convert_by_vocab(vocab, items):
		"""Converts a sequence of [tokens|ids] using the vocab."""
		output = []
		for item in items:
			output.append(vocab[item])
		return output
	return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
	"""Runs basic whitespace cleaning and splitting on a piece of text."""
	text = text.strip()
	if not text:
		return []
	tokens = text.split()
	return tokens

unused_token = {}
for i in range(1, 100):
    unused_token['[unused{}]'.format(i)] = i

class FullTokenizer(object):
	"""Runs end-to-end tokenziation."""
	def __init__(self, vocab_file, do_lower_case=True, do_whole_word_mask=False):
		self.vocab = load_vocab(vocab_file)
		self.inv_vocab = {v: k for k, v in self.vocab.items()}
		self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, 
												do_whole_word_mask=do_whole_word_mask)
		self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

	def tokenize(self, text):
		split_tokens = []
		for token in self.basic_tokenizer.tokenize(text):
			if token in unused_token:
				split_tokens.append(token)
				continue
			for sub_token in self.wordpiece_tokenizer.tokenize(token):
				split_tokens.append(sub_token)

		return split_tokens

	def convert_tokens_to_ids(self, tokens, max_length=None):
		return convert_tokens_to_ids(self.vocab, tokens)

	def convert_ids_to_tokens(self, ids):
		return convert_ids_to_tokens(self.inv_vocab, ids)

	def covert_tokens_to_char_ids(self, tokens, max_length=None, char_len=5):
		pass

	def padding(self, token_id_lst, max_length, zero_padding=0):
		return token_id_lst + [zero_padding] * (max_length - len(token_id_lst))
		
class BasicTokenizer(object):
	"""Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

	def __init__(self, do_lower_case=True, do_whole_word_mask=False):
		"""Constructs a BasicTokenizer.
		Args:
		  do_lower_case: Whether to lower case the input.
		"""
		self.do_lower_case = do_lower_case
		self.do_whole_word_mask = do_whole_word_mask

	def tokenize(self, text):
		"""Tokenizes a piece of text."""
		text = convert_to_unicode(text)
		text = self._clean_text(text)

		# This was added on November 1st, 2018 for the multilingual and Chinese
		# models. This is also applied to the English models now, but it doesn't
		# matter since the English models were not trained on any Chinese data
		# and generally don't have any Chinese data in them (there are Chinese
		# characters in the vocabulary because Wikipedia does have some Chinese
		# words in the English Wikipedia.).
		if not self.do_whole_word_mask:
			text = self._tokenize_chinese_chars(text)

		orig_tokens = whitespace_tokenize(text)
		split_tokens = []
		for token in orig_tokens:
			if self.do_lower_case:
				token = token.lower()
				token = self._run_strip_accents(token)
			if token in unused_token:
				split_tokens.append(token)
				continue
			split_tokens.extend(self._run_split_on_punc(token))

		output_tokens = whitespace_tokenize(" ".join(split_tokens))
		return output_tokens

	def _run_strip_accents(self, text):
		"""Strips accents from a piece of text."""
		text = unicodedata.normalize("NFD", text)
		output = []
		for char in text:
			cat = unicodedata.category(char)
			if cat == "Mn":
				continue
			output.append(char)
		return "".join(output)

	def _run_split_on_punc(self, text):
		"""Splits punctuation on a piece of text."""
		chars = list(text)
		i = 0
		start_new_word = True
		output = []
		while i < len(chars):
			char = chars[i]
			if _is_punctuation(char):
				output.append([char])
				start_new_word = True
			else:
				if start_new_word:
					output.append([])
				start_new_word = False
				output[-1].append(char)
			i += 1

		return ["".join(x) for x in output]

	def _tokenize_chinese_chars(self, text):
		"""Adds whitespace around any CJK character."""
		output = []
		for char in text:
			cp = ord(char)
			if self._is_chinese_char(cp):
				output.append(" ")
				output.append(char)
				output.append(" ")
			else:
				output.append(char)
		return "".join(output)

	def _is_chinese_char(self, cp):
		"""Checks whether CP is the codepoint of a CJK character."""
		# This defines a "chinese character" as anything in the CJK Unicode block:
		#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
		#
		# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
		# despite its name. The modern Korean Hangul alphabet is a different block,
		# as is Japanese Hiragana and Katakana. Those alphabets are used to write
		# space-separated words, so they are not treated specially and handled
		# like the all of the other languages.
		if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
			(cp >= 0x3400 and cp <= 0x4DBF) or  #
			(cp >= 0x20000 and cp <= 0x2A6DF) or  #
			(cp >= 0x2A700 and cp <= 0x2B73F) or  #
			(cp >= 0x2B740 and cp <= 0x2B81F) or  #
			(cp >= 0x2B820 and cp <= 0x2CEAF) or
			(cp >= 0xF900 and cp <= 0xFAFF) or  #
			(cp >= 0x2F800 and cp <= 0x2FA1F)):  #
			return True

		return False

	def _clean_text(self, text):
		"""Performs invalid character removal and whitespace cleanup on text."""
		output = []
		for char in text:
			cp = ord(char)
			if cp == 0 or cp == 0xfffd or _is_control(char):
				continue
			if _is_whitespace(char):
				output.append(" ")
			else:
				output.append(char)
		return "".join(output)


class WordpieceTokenizer(object):
	"""Runs WordPiece tokenziation."""

	def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
		self.vocab = vocab
		self.unk_token = unk_token
		self.max_input_chars_per_word = max_input_chars_per_word

	def tokenize(self, text):
		"""Tokenizes a piece of text into its word pieces.
		This uses a greedy longest-match-first algorithm to perform tokenization
		using the given vocabulary.
		For example:
		  input = "unaffable"
		  output = ["un", "##aff", "##able"]
		Args:
		  text: A single token or whitespace separated tokens. This should have
			already been passed through `BasicTokenizer.
		Returns:
		  A list of wordpiece tokens.
		"""

		text = convert_to_unicode(text)

		output_tokens = []
		for token in whitespace_tokenize(text):
			chars = list(token)
			if len(chars) > self.max_input_chars_per_word:
				output_tokens.append(self.unk_token)
				continue

			is_bad = False
			start = 0
			sub_tokens = []
			while start < len(chars):
				end = len(chars)
				cur_substr = None
				while start < end:
					substr = "".join(chars[start:end])
					if start > 0:
						substr = "##" + substr
					if substr in self.vocab:
						cur_substr = substr
						break
					end -= 1
				if cur_substr is None:
					is_bad = True
					break
				sub_tokens.append(cur_substr)
				start = end

			if is_bad:
				output_tokens.append(self.unk_token)
			else:
				output_tokens.extend(sub_tokens)
		return output_tokens

def _is_chinese_char(cp):
	"""Checks whether CP is the codepoint of a CJK character."""
	# This defines a "chinese character" as anything in the CJK Unicode block:
	#   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
	#
	# Note that the CJK Unicode block is NOT all Japanese and Korean characters,
	# despite its name. The modern Korean Hangul alphabet is a different block,
	# as is Japanese Hiragana and Katakana. Those alphabets are used to write
	# space-separated words, so they are not treated specially and handled
	# like the all of the other languages.
	if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
		(cp >= 0x3400 and cp <= 0x4DBF) or  #
		(cp >= 0x20000 and cp <= 0x2A6DF) or  #
		(cp >= 0x2A700 and cp <= 0x2B73F) or  #
		(cp >= 0x2B740 and cp <= 0x2B81F) or  #
		(cp >= 0x2B820 and cp <= 0x2CEAF) or
		(cp >= 0xF900 and cp <= 0xFAFF) or  #
		(cp >= 0x2F800 and cp <= 0x2FA1F)):  #
		return True

	return False

def _is_whitespace(char):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically contorl characters but we treat them
	# as whitespace since they are generally considered as such.
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False


def _is_control(char):
	"""Checks whether `chars` is a control character."""
	# These are technically control characters but we count them as whitespace
	# characters.
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat.startswith("C"):
		return True
	return False


def _is_punctuation(char):
	"""Checks whether `chars` is a punctuation character."""
	cp = ord(char)
	# We treat all non-letter/number ASCII as punctuation.
	# Characters such as "^", "$", and "`" are not in the Unicode
	# Punctuation class but we treat them as punctuation anyways, for
	# consistency.
	if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
	  (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
		return True
	cat = unicodedata.category(char)
	if cat.startswith("P"):
		return True
	return False

tokenizer = FullTokenizer(
			vocab_file='../kernel/vocab.txt', 
			do_lower_case=False,
			do_whole_word_mask=False)

import jieba
for word in unused_token:
    jieba.add_word(word)
print(tokenizer.tokenize(" ".join(jieba.cut('罗杰·加西亚和贝拉乌桑几乎[unused1]的情况下'))))


idiom_dict = {}
length = []
left = []
import json, codecs
with codecs.open('../kernel/idiomDict_clean.json', 'r', 'utf-8') as frobj:
    for i, line in enumerate(frobj):
        content = json.loads(line.strip())
        if isinstance(content['info'], list):
            idiom_dict[content['word']] = "。".join(content['info'])
        else:
            idiom_dict[content['word']] = content['info']
        length.append(len(idiom_dict[content['word']]))
        if len(idiom_dict[content['word']]) >= 64:
            left.append(content['word'])

import jieba
import copy, json

import re
answer_pattern = re.compile('#idiom\d+#')
postprocess_answer_pattern = re.compile('\[unused\d+\]')

from collections import namedtuple
max_length = 340
answer_length = 70
max_label = 5
input_lst = []

unused_token = {}
for i in range(1, 100):
    unused_token['[unused{}]'.format(i)] = i

from itertools import accumulate

import random

def cut_doc(tokens_a_id, answer_symbol_id_pos, answer_symbol, max_length):
        
    before_part = tokens_a_id[0:answer_symbol_id_pos]
    after_part = tokens_a_id[answer_symbol_id_pos:]
    
    half_length = int(max_length / 2)
    if len(before_part) < half_length: # cut at tail
        st = 0
        ed = min(len(before_part) + 1 + len(after_part), max_length - 3)
    elif len(after_part) < half_length: # cut at head
        ed = len(before_part) + 1 + len(after_part)
        st = max(0, ed - (max_length - 3))
    else: # cut at both sides
        st = len(before_part) + 3 - half_length
        ed = len(before_part) + half_length
    
    output = tokens_a_id[st:ed]
    assert tokens_a_id[answer_symbol_id_pos] in output
    return output

unused_id = [tokenizer.vocab[unused_word] for unused_word in list(unused_token.keys())]
def replace_unused2mask(tokens_a_id, unused_id, mask_id):
    for i, ids in enumerate(tokens_a_id):
        if ids in unused_id:
            tokens_a_id[i] = mask_id
    return tokens_a_id

def get_unused(tokens_a_id, unused_id):
    unused_id_pos = []
    for i, ids in enumerate(tokens_a_id):
        if ids in unused_id:
            unused_id_pos.append(ids)
    return unused_id_pos
            
def form_choice(input_dict, label_positions, context_index, max_length):
    tokens_a_id = input_dict['tokens_a']
    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    mask_id = tokenizer.vocab['[MASK]']
    output_dict_lst = []
    
    unused_id_pos = get_unused(tokens_a_id, unused_id)
    tokens_a_id = replace_unused2mask(tokens_a_id, unused_id, mask_id)
    

    for item in input_dict["ans"]:
        tmp = {}
        tokens_b_id = item['tokens_b']
        segmend_id = [0]+[0]*len(tokens_a_id)+[0]+[1]*len(tokens_b_id)+[1]
        tokens_id = [cls_id]+tokens_a_id+[sep_id]+tokens_b_id+[sep_id]
        assert len(segmend_id) == len(tokens_id)
        tokens_id = tokenizer.padding(tokens_id, max_length, 0)
        segment_id = tokenizer.padding(segmend_id, max_length, 0)
        assert len(tokens_id) == max_length
        assert len(segment_id) == max_length
        labels = item['label_ids'] + [0]*(5-len(item['label_positions']))
        label_positions = [pos+1 for pos in item['label_positions']] + [0]*(5-len(item['label_positions']))
        label_weights = [1]*len(item['label_positions']) + [0]*(5-len(item['label_positions']))
        tmp['input_ids'] = tokens_id
        tmp['context_id'] = context_index
        tmp['segment_ids'] = segment_id
        tmp['label_positions'] = label_positions
        tmp['label_ids'] = labels
        tmp['label_weights'] = label_weights
        tmp['tokens'] = item['tokens']
        tmp['answer_tokens'] = item['answer_tokens']
        output_dict_lst.append(tmp)
    return output_dict_lst

def dev_cut_postprocess(ans_tokens_a_id, old_answer_lst, candidate, context_index, max_length, 
                       if_detail=False):
    context = " ".join(tokenizer.convert_ids_to_tokens(ans_tokens_a_id))
    answer_lst = []
    answer_token_dict = {}
    for item in old_answer_lst:
        answer_token_dict[item[0]] = item
    for i, l in enumerate(postprocess_answer_pattern.finditer(context)):
        if l.group() in answer_token_dict:
            answer_lst.append(answer_token_dict[l.group()])
                    
    tokens_a_id = ans_tokens_a_id
    label_positions = []
    for answer in answer_lst:
        label_positions.append(tokens_a_id.index(unused_token[answer[0]]))
    label_positions.sort()

    label_positions_dict = {}
    for answer in answer_lst:
        pos = tokens_a_id.index(unused_token[answer[0]])
        label_positions_dict[pos] = answer
    
    answers = [item[1] for item in answer_lst]
                                
    answer_lst_dict = {}
    for answer in answer_lst:
        answer_lst_dict[answer[-1]] = answer
            
    tmp = {}
    tmp['ans'] = []
    tmp['tokens_a'] = tokens_a_id

    for other in candidate:
        if not if_detail:
            answer_token = other
        else:
            answer_token = other + "。" + idiom_dict.get(other, "")
        tokens_b = tokenizer.tokenize(" ".join(jieba.cut(answer_token)))
        tokens_b_id = tokenizer.convert_tokens_to_ids(tokens_b)
        sub_tmp = {}
        sub_tmp['label_ids'] = [0]*len(label_positions)
        sub_tmp['tokens_b'] = tokens_b_id
        sub_tmp['label_positions'] = label_positions
        sub_tmp['answer_tokens'] = answer_token
        sub_tmp['tokens'] = label_positions_dict
        tmp['ans'].append(sub_tmp)
        
    return form_choice(tmp, label_positions, context_index, max_length)

answer_pattern = re.compile('#idiom\d+#')
postprocess_answer_pattern = re.compile('\[unused\d+\]')

# def process_dev(input_path):
with codecs.open('../kernel/dev.txt', 'r', 'utf-8') as frobj:
    context_index = 0
    dev_total_output = []
    flag = False
    total_cnt = 0
    for line_index, line in enumerate(frobj):
        flag = False
        content = json.loads(line.strip())
        candidate = content['candidates']
        context = content['content']
        
        context_output = []
        total_answer_lst = []

        context_cnt = 0

        for sub_context in context:
            sub_context_output = []
            answer_lst = []
            for i, l in enumerate(answer_pattern.finditer(sub_context)):
                answer_lst.append(["[unused{}]".format(i+1), l.group()])
                total_answer_lst.append(["[unused{}]".format(i+1+context_cnt), l.group()])
                sub_context = re.sub(l.group(), "[unused{}]".format(i+1), sub_context)
            context_cnt += len(answer_lst)

            tokens_a = tokenizer.tokenize(" ".join(jieba.cut(sub_context)))
            tokens_a_id = tokenizer.convert_tokens_to_ids(tokens_a)

            label_positions = []
            for answer in answer_lst:
                label_positions.append(tokens_a_id.index(unused_token[answer[0]]))
            label_positions.sort()

            label_positions_dict = {}
            for index, pos in enumerate(label_positions):
                label_positions_dict[pos] = index

            tmp = {}
            tmp['ans'] = []
            id_dict = {}
            answers = [item[1] for item in answer_lst]
            for answer in answer_lst:
                ans_tokens_a_id = cut_doc(tokens_a_id, 
                                          tokens_a_id.index(unused_token[answer[0]]),
                                         answer[0],
                                         max_length-answer_length)
                if " ".join([str(idx) for idx in ans_tokens_a_id]) in id_dict:
                    continue
                else:
                    id_dict[" ".join([str(idx) for idx in ans_tokens_a_id])] = 0
                resp = dev_cut_postprocess(ans_tokens_a_id, answer_lst, candidate, context_index, max_length,
                                          if_detail=True)
                sub_context_output.extend(resp)
            context_output.append(sub_context_output)
        dev_total_output.append(context_output)

from functools import reduce
import numpy as np
import json
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import os
import time

def deleteDuplicate_v1(input_dict_lst):
	f = lambda x,y:x if y in x else x + [y]
	return reduce(f, [[], ] + input_dict_lst)

def get_context_pair(resp, l):
	label_weights = l['label_weights']
	valid_resp = {}
	for key in resp:
		valid_resp[key] = []
		for index, value in enumerate(resp[key]):
			if label_weights[index] == 1:
				valid_resp[key].append(value)

	answer = l['answer_tokens']
	position_tokens = l['tokens']
	label_position = [lpos-1 for index, lpos in enumerate(l['label_positions']) if label_weights[index]==1]
	
	score_label = []
	for index in range(len(valid_resp['pred_label'])):
		label = valid_resp['pred_label'][index]
		score = valid_resp['max_prob'][index]
		position = label_position[index]
		position_token = position_tokens[str(position)][1]
		if label == 1:
			score = 1 - score
		score_label.append({"score":score, "label":label, 
							"position_token":position_token,
						   "answer":answer})
	return score_label

def format_socre_matrix(result_lst, score_merge='mean'):
	answer_dict = {}
	candidate_dict = {}
	answer_index = 0
	pos_index = 0
	for item in result_lst:
		if item['answer'] not in answer_dict:
			answer_dict[item['answer']] = answer_index
			answer_index += 1
		if item['position_token'] not in candidate_dict:
			candidate_dict[item['position_token']] = pos_index
			pos_index += 1
			
	score_matrix = -np.ones((len(answer_dict), len(candidate_dict)))
	for item in result_lst:
		answer_pos = answer_dict[item['answer']]
		candidate_pos = candidate_dict[item['position_token']]
		score_matrix_score = score_matrix[answer_pos, candidate_pos]
		if score_matrix_score == -1:
			score_matrix[answer_pos, candidate_pos] = item['score']
		else:
			if score_merge == 'mean':
				score_matrix[answer_pos, candidate_pos] += item['score']
				score_matrix[answer_pos, candidate_pos] /= 2
			elif score_merge == 'max':
				if item['score'] > score_matrix[answer_pos, candidate_pos]:
					score_matrix[answer_pos, candidate_pos] = item['score']
	return score_matrix, answer_dict, candidate_dict

from tensorflow.contrib import predictor
model_dict = {
    "model":model_file
}
chid_model = predictor.from_saved_model(model_dict['model'])

cnt = 0
candidate_name_dict = {}
valid_keys = ['input_ids', 'label_weights', 
			'label_positions', 'label_ids',
			'segment_ids']
for index, line in enumerate(f):
    content = json.loads(line.strip())
    total_resp = []
    start = time.time()
    for t in content:
        tmp = {}
        for l in t:
            for key in valid_keys:
                if key in tmp:
                    tmp[key].append(l[key])
                else:
                    tmp[key] = [l[key]]
            # tmp = {
            # 	"input_ids":np.array([l['input_ids']]),
            # 	'label_weights':np.array([l['label_weights']]),
            # 	'label_positions':np.array([l['label_positions']]),
            # 	'label_ids':np.array([l['label_ids']]),
            # 	'segment_ids':np.array([l['segment_ids']]),
            # }
        resp = chid_model(tmp)
        resp_lst = []
        batch_size = int(resp['pred_label'].shape[0]/5)
        for key in resp:
            resp[key] = np.reshape(resp[key], [-1, 5]).tolist()

        for i_index in range(batch_size):
            tmp = {
                "pred_label":resp['pred_label'][i_index],
                "max_prob":resp['max_prob'][i_index],
            }
            resp_lst.append(tmp)
        for i_index in range(len(t)):
            resp_ = resp_lst[i_index]
            l_ = t[i_index]
            result = get_context_pair(resp_, l_)
            total_resp.extend(result)
    total_resp = deleteDuplicate_v1(total_resp)
    resp = format_socre_matrix(total_resp, score_merge=FLAGS.score_merge)
    row_ind, col_ind = linear_sum_assignment(resp[0])
    mapping_dict = dict(zip(col_ind, row_ind))
    dura = time.time()-start
    candidte_dict = resp[-1]
    candidate_inverse_dict = {}
    for key in candidte_dict:
        candidate_inverse_dict[candidte_dict[key]] = key

    for col in mapping_dict:
        col_name = candidate_inverse_dict[col]
        candidate_name_dict[col_name] = int(mapping_dict[col])
    if np.mod(index, 100) == 0:
        print(index, dura)

from collections import OrderedDict
import pandas as pd
total_result_mask = []
total_result_ans = []
for key in total_result:
    total_result_mask.append(key)
    total_result_ans.append(total_result[key])
df = pd.DataFrame(OrderedDict({
    "mask":total_result_mask,
    "ans":total_result_ans
}))
pd.to_csv('submission.csv', 
          header=False,
             index=False)