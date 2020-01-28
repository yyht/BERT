import tensorflow as tf

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self, label_id):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	def _read_data(self, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		raise NotImplementedError()