import tensorflow as tf
from data_generator import tokenization

class InputExample(object):
	def __init__(self, qas_id,
							 question_text,
							 doc_tokens,
							 orig_answer_text=None,
							 start_position=None,
							 end_position=None,
							 answer_choice=None,
							 choice=None,
							 input_shape=None):
		self.qas_id = qas_id
		self.question_text = question_text
		self.doc_tokens = doc_tokens
		self.orig_answer_text = orig_answer_text
		self.start_position = start_position
		self.end_position = end_position
		self.answer_choice = answer_choice # list of string with choice order
		self.choice = choice
		self.input_shape = input_shape

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		s = ""
		s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
		s += ", question_text: %s" % (
				tokenization.printable_text(self.question_text))
		s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
		if self.start_position:
			s += ", start_position: %d" % (self.start_position)
		if self.end_position:
			s += ", end_position: %d" % (self.end_position)
		if self.answer_choice:
			for index, answer in enumerate(self.answer_choice):
				s += ", answer_choice: {} {}".format(index, answer)
		if self.choice:
			s += ", correct choice: {}".format(self.choice)
		return s

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self,
							 unique_id,
							 example_index=None,
							 doc_span_index=None,
							 tokens=None,
							 token_to_orig_map=None,
							 token_is_max_context=None,
							 input_ids=None,
							 input_mask=None,
							 segment_ids=None,
							 start_position=None,
							 end_position=None,
							 answer_choice=None,
							 choice=None):
		self.unique_id = unique_id
		self.example_index = example_index
		self.doc_span_index = doc_span_index
		self.tokens = tokens
		self.token_to_orig_map = token_to_orig_map
		self.token_is_max_context = token_is_max_context
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.start_position = start_position
		self.end_position = end_position
		self.choice = choice
		self.answer_choice = answer_choice