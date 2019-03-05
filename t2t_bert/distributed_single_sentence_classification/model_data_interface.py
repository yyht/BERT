import tensorflow as tf

def data_interface(FLAGS):
	if FLAGS.model_type in ["bert", "bert_rule"]:
		name_to_features = {
				"input_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
		}
	elif FLAGS.model_type in ["textcnn"]:
		name_to_features = {
			"input_ids_a":tf.FixedLenFeature([FLAGS.max_length], tf.int64),
			"label_ids":tf.FixedLenFeature([], tf.int64)
		}
		if FLAGS.with_char == "char":
			name_to_features["input_char_ids_a"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)

	return name_to_features
