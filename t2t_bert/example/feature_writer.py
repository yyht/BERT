import tensorflow as tf
from data_generator import tf_data_utils
import collections

class FeatureWriter(object):
	def __init__(self, filename, is_training):
		self.filename = filename
		self.is_training = is_training
		self.num_features = 0
		self._writer = tf.python_io.TFRecordWriter(filename)
	def process_feature(self, feature):
		raise NotImplementedError()
	def close(self):
		self._writer.close() 

class SpanFeatureWriter(FeatureWriter):
	"""Writes InputFeature to TF example file."""

	def __init__(self, filename, is_training):
		super(SpanFeatureWriter, self).__init__(filename, is_training)

	def process_feature(self, feature):
		"""Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
		self.num_features += 1

		features = collections.OrderedDict()
		features["unique_ids"] = tf_data_utils.create_int_feature([feature.unique_id])
		features["input_ids"] = tf_data_utils.create_int_feature(feature.input_ids)
		features["input_mask"] = tf_data_utils.create_int_feature(feature.input_mask)
		features["segment_ids"] = tf_data_utils.create_int_feature(feature.segment_ids)

		if self.is_training:
			features["start_positions"] = tf_data_utils.create_int_feature([feature.start_position])
			features["end_positions"] = tf_data_utils.create_int_feature([feature.end_position])

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		self._writer.write(tf_example.SerializeToString())

class ClassifierFeatureWriter(FeatureWriter):
	def __init__(self, filename, is_training):
		super(ClassifierFeatureWriter, self).__init__(filename, is_training)

	def process_feature(self, feature):
		"""Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
		self.num_features += 1

		features = collections.OrderedDict()
		features["input_ids"] = tf_data_utils.create_int_feature(feature.input_ids)
		features["input_mask"] = tf_data_utils.create_int_feature(feature.input_mask)
		features["segment_ids"] = tf_data_utils.create_int_feature(feature.segment_ids)
		features["label_ids"] = tf_data_utils.create_int_feature([feature.label_ids])
		try:
			features["qas_id"] = tf_data_utils.create_int_feature([feature.guid])
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())
		except:
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())

class PairClassifierFeatureWriter(FeatureWriter):
	def __init__(self, filename, is_training):
		super(PairClassifierFeatureWriter, self).__init__(filename, is_training)

	def process_feature(self, feature):
		"""Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
		self.num_features += 1

		features = collections.OrderedDict()
		features["input_ids_a"] = tf_data_utils.create_int_feature(feature.input_ids_a)
		features["input_mask_a"] = tf_data_utils.create_int_feature(feature.input_mask_a)
		features["segment_ids_a"] = tf_data_utils.create_int_feature(feature.segment_ids_a)
		features["input_ids_b"] = tf_data_utils.create_int_feature(feature.input_ids_b)
		features["input_mask_b"] = tf_data_utils.create_int_feature(feature.input_mask_b)
		features["segment_ids_b"] = tf_data_utils.create_int_feature(feature.segment_ids_b)
		features["label_ids"] = tf_data_utils.create_int_feature([feature.label_ids])
		try:
			features["qas_id"] = tf_data_utils.create_int_feature([feature.guid])
		except:
			pass
		try:
			features["class_ratio"] = tf_data_utils.create_float_feature([feature.class_ratio])
		except:
			pass

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		self._writer.write(tf_example.SerializeToString())
		

class MultiChoiceFeatureWriter(FeatureWriter):
	def __init__(self, filename, is_training):
		super(MultiChoiceFeatureWriter, self).__init__(filename, is_training)

	def process_feature(self, feature):
		"""Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
		self.num_features += 1

		features = collections.OrderedDict()
		features["input_ids"] = tf_data_utils.create_int_feature(feature.input_ids)
		features["input_mask"] = tf_data_utils.create_int_feature(feature.input_mask)
		features["segment_ids"] = tf_data_utils.create_int_feature(feature.segment_ids)
		features["label_ids"] = tf_data_utils.create_int_feature([feature.choice])

		try:
			features["qas_id"] = tf_data_utils.create_int_feature([feature.unique_id])
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())
		except:
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())

class NormalEncoderFeatureWriter(FeatureWriter):
	def __init__(self, filename, is_training):
		super(NormalEncoderFeatureWriter, self).__init__(filename, is_training)

	def process_feature(self, feature):
		self.num_features += 1
		features = collections.OrderedDict()

		features["input_ids_a"] = tf_data_utils.create_int_feature(feature.input_ids_a)
		features["label_ids"] = tf_data_utils.create_int_feature([feature.label_ids])

		try:
			features["input_char_ids_a"] = tf_data_utils.create_int_feature([feature.input_char_ids_a])
		except:
			print("==not apply char==")
		try:
			features["input_ids_b"] = tf_data_utils.create_int_feature([feature.input_ids_b])
		except:
			print("==not use second sentence==")
		try:
			features["input_char_ids_b"] = tf_data_utils.create_int_feature([feature.input_char_ids_b])
		except:
			print("==not use second sentence==")
		try:
			features["label_probs"] = tf_data_utils.create_float_feature([feature.label_probs])
		except:
			print("==not use soft labels==")

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())

class PairPreTrainingFeature(FeatureWriter):
	def __init__(self, filename, is_training):
		super(PairPreTrainingFeature, self).__init__(filename, is_training)

	def process_feature(self, feature):
		self.num_features += 1
		features = collections.OrderedDict()

		features["input_ids"] = tf_data_utils.create_int_feature(feature.input_ids)
		features["input_mask"] = tf_data_utils.create_int_feature(feature.input_mask)
		features["segment_ids"] = tf_data_utils.create_int_feature(feature.segment_ids)
		features["masked_lm_positions"] = tf_data_utils.create_int_feature(feature.masked_lm_positions)
		features["masked_lm_ids"] = tf_data_utils.create_int_feature(feature.masked_lm_ids)
		features["masked_lm_weights"] = tf_data_utils.create_float_feature(feature.masked_lm_weights)
		features["label_ids"] = tf_data_utils.create_int_feature([feature.label_ids])

		try:
			features["qas_id"] = tf_data_utils.create_int_feature([feature.guid])
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())
		except:
			tf_example = tf.train.Example(features=tf.train.Features(feature=features))
			self._writer.write(tf_example.SerializeToString())