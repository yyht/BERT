import tensorflow as tf

def data_interface(FLAGS):
	if FLAGS.model_type in ["bert", "bert_rule", "bert_small"]:
		if FLAGS.task_type == "single_sentence_classification":
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
		elif FLAGS.task_type == "pair_sentence_classification":
			name_to_features = {
				"input_ids_a":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask_a":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids_a":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_ids_b":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask_b":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids_b":
						tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"label_ids":
						tf.FixedLenFeature([], tf.int64),
				}
	elif FLAGS.model_type in ["textcnn", "textlstm"]:
		name_to_features = {
			"input_ids_a":tf.FixedLenFeature([FLAGS.max_length], tf.int64),
			"label_ids":tf.FixedLenFeature([], tf.int64)
		}
		if FLAGS.with_char == "char":
			name_to_features["input_char_ids_a"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)
			if FLAGS.task_type == "pair_sentence_classification":
				name_to_features["input_char_ids_b"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)
		if FLAGS.task_type == "pair_sentence_classification":
			name_to_features["input_ids_b"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)

	elif FLAGS.model_type in ["textcnn_distillation", "textlstm_distillation"]:
		name_to_features = {
			"input_ids_a":tf.FixedLenFeature([FLAGS.max_length], tf.int64),
			"label_ids":tf.FixedLenFeature([], tf.int64),
			"label_ratio":tf.FixedLenFeature([], tf.float32),
			"label_probs":tf.FixedLenFeature([FLAGS.num_classes], tf.float32),
			"distillation_ratio":tf.FixedLenFeature([], tf.float32)
		}
		if FLAGS.with_char == "char":
			name_to_features["input_char_ids_a"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)
			if FLAGS.task_type == "pair_sentence_classification":
				name_to_features["input_char_ids_b"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)
		if FLAGS.task_type == "pair_sentence_classification":
			name_to_features["input_ids_b"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)

	return name_to_features
