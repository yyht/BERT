import tensorflow as tf
from data_generator import tokenization

class PreTrainingInstance(object):
	"""A single training instance (sentence pair)."""

	def __init__(self, guid, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
				is_random_next, label):
		self.guid = guid
		self.tokens = tokens
		self.segment_ids = segment_ids
		self.is_random_next = is_random_next
		self.masked_lm_positions = masked_lm_positions
		self.masked_lm_labels = masked_lm_labels
		self.label = label

	def __str__(self):
		s = ""
		s += "guid: {}\n".format(self.guid)
		s += "tokens: %s\n" % (" ".join(
			[tokenization.printable_text(x) for x in self.tokens]))
		s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
		s += "is_random_next: %s\n" % self.is_random_next
		s += "masked_lm_positions: %s\n" % (" ".join(
			[str(x) for x in self.masked_lm_positions]))
		s += "masked_lm_labels: %s\n" % (" ".join(
			[tokenization.printable_text(x) for x in self.masked_lm_labels]))
		s += "label: {}\n".format(self.label)
		s += "\n"
		return s

	def __repr__(self):
		return self.__str__()

class PreTrainingFeature(object):
	"""A single set of features of data."""

	def __init__(self, guid, 
		input_ids, input_mask, segment_ids,
		masked_lm_positions, masked_lm_ids,
		masked_lm_weights,
		label_ids, is_random_next):
		
		self.guid = guid

		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids

		self.masked_lm_positions = masked_lm_positions
		self.masked_lm_ids = masked_lm_ids
		self.masked_lm_weights = masked_lm_weights

		self.label_ids = label_ids
		self.is_random_next = is_random_next