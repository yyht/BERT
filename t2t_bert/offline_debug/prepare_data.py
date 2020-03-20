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
			vocab_file='./data/chinese_L-12_H-768_A-12/vocab.txt', 
			do_lower_case=True,
			do_whole_word_mask=False)


import random

def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

def split_on_uppercase(s, keep_contiguous=True):
    """

    Args:
        s (str): string
        keep_contiguous (bool): flag to indicate we want to 
                                keep contiguous uppercase chars together

    Returns:

    """

    string_length = len(s)
    is_lower_around = (lambda: s[i-1].islower() or 
                       string_length > (i + 1) and s[i + 1].islower())

    start = 0
    parts = []
    for i in range(1, string_length):
        if s[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(s[start: i])
            start = i
    parts.append(s[start:])

    return parts

import os,sys, json, re
train_writer = tf.python_io.TFRecordWriter('/data/xuht/data0310_shuffled_data_yuxue_train.tfrecord')
dev_writer = tf.python_io.TFRecordWriter('/data/xuht/data0310_shuffled_data_yuxue_dev.tfrecord')
total_num = 192438
max_length = 2048
with open('/data/xuht/data0310_shuffle.txt', 'r') as frobj:
    label_lst = []
    for index, line in enumerate(frobj):
        content = json.loads(line.strip())
        label = content['label']
        output = []
        string_lst = split_on_uppercase(content['data'], True)
        context = re.sub("[\r\n]+", " ", " ".join(string_lst).lower())
        tokens_a = tokenizer.tokenize(context)
#         print(tokens_a, len(tokens_a))
#         break
        if index == 0:
            print(tokens_a, context, content['data'], len(tokens_a))
        if len(tokens_a) >= max_length:
            tokens_a = tokens_a[0:max_length]
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a, max_length)
        input_ids_a = tokenizer.padding(input_ids_a, max_length, 0)
        
        assert len(input_ids_a) == max_length
        
        features = {
                "input_ids_a":create_int_feature(input_ids_a)
            }
        if label:
            features['label_ids'] = create_int_feature([1])
        else:
            features['label_ids'] = create_int_feature([0])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        if index <= int(total_num*0.8):
            train_writer.write(tf_example.SerializeToString())
        else:
            dev_writer.write(tf_example.SerializeToString())
train_writer.close()
dev_writer.close()