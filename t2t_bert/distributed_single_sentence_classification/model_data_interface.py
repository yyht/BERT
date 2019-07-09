import tensorflow as tf

def data_interface(FLAGS):
	if FLAGS.model_type in ["bert","bert_small"]:
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
		elif FLAGS.task_type in ['bert_pretrain']:
			name_to_features = {
				"input_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"input_mask":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"segment_ids":
					tf.FixedLenFeature([FLAGS.max_length], tf.int64),
				"masked_lm_positions":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
					tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
					tf.FixedLenFeature([], tf.int64),
				}

	if FLAGS.model_type in ["bert_rule"]:
		if FLAGS.task_type == "single_sentence_classification":
			name_to_features = {
					"input_ids":
							tf.FixedLenFeature([FLAGS.max_length], tf.int64),
					"input_mask":
							tf.FixedLenFeature([FLAGS.max_length], tf.int64),
					"segment_ids":
							tf.FixedLenFeature([FLAGS.max_length], tf.int64),
					"rule_ids":
							tf.FixedLenFeature([FLAGS.max_length], tf.int64),
					"label_ids":
							tf.FixedLenFeature([], tf.int64),
			}

	elif FLAGS.model_type in ["textcnn", "textlstm", "dan"]:
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

	elif FLAGS.model_type in ["textcnn_distillation", "textlstm_distillation", "dan_distillation"]:
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
		if FLAGS.distillation in ["feature_distillation", "mdd_distillation", "rkd_distillation"]:
			name_to_features["distillation_feature"] = tf.FixedLenFeature([768], tf.float32)

	elif FLAGS.model_type in ["textcnn_distillation_adv_adaptation"]:
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
		if FLAGS.distillation in ["feature_distillation", "mdd_distillation", "rkd_distillation"]:
			name_to_features["distillation_feature"] = tf.FixedLenFeature([768], tf.float32)

		if FLAGS.distillation in ['adv_adaptation_distillation']:
			name_to_features['adv_ids'] = tf.FixedLenFeature([], tf.int64)

	elif FLAGS.model_type in ["match_pyramid", "match_pyramid_distillation"]:
		name_to_features = {
			"input_ids_a":tf.FixedLenFeature([FLAGS.max_length], tf.int64),
			"input_ids_b":tf.FixedLenFeature([FLAGS.max_length], tf.int64),
			"label_ids":tf.FixedLenFeature([], tf.int64),
			"label_ratio":tf.FixedLenFeature([], tf.float32),
			"label_probs":tf.FixedLenFeature([FLAGS.num_classes], tf.float32),
			"distillation_ratio":tf.FixedLenFeature([], tf.float32)
		}
		if FLAGS.distillation in ["feature_distillation", "mdd_distillation", "rkd_distillation"]:
			name_to_features["distillation_feature"] = tf.FixedLenFeature([768], tf.float32) 
		if FLAGS.with_char == "char":
			name_to_features["input_char_ids_a"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)
			name_to_features["input_char_ids_b"] = tf.FixedLenFeature([FLAGS.max_length], tf.int64)

	return name_to_features

def data_interface_server(FLAGS):
	if FLAGS.model_type in ["bert", "bert_rule", "bert_small"]:
		if FLAGS.task_type == "single_sentence_classification":

			receiver_tensors = {
				"input_ids":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids'),
				"input_mask":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_mask'),
				"segment_ids":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='segment_ids'),
				"label_ids":
						tf.placeholder(tf.int32, [None], name='label_ids'),
			}

		elif FLAGS.task_type == "pair_sentence_classification":

			receiver_tensors = {
				"input_ids_a":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_a'),
				"input_mask_a":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_mask_a'),
				"segment_ids_a":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='segment_ids_a'),
				"input_ids_b":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_b'),
				"input_mask_b":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_mask_b'),
				"segment_ids_b":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='segment_ids_b'),
				"label_ids":
						tf.placeholder(tf.int32, [None], name='label_ids'),
			}

	elif FLAGS.model_type in ["textcnn", "textlstm", "dan"]:
		receiver_tensors = {
				"input_ids_a":
						tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_a'),
				"label_ids":
						tf.placeholder(tf.int32, [None], name='label_ids')
			}

		if FLAGS.with_char == "char":
			receiver_tensors["input_char_ids_a"] = tf.placeholder(tf.int32, [None, FLAGS.char_limit, FLAGS.max_length], name='input_char_ids_a')
				
		if FLAGS.task_type == "pair_sentence_classification":
			receiver_tensors["input_ids_b"] = tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_b')
			if FLAGS.with_char == "char":
				receiver_tensors["input_char_ids_b"] = tf.placeholder(tf.int32, [None, FLAGS.char_limit, FLAGS.max_length], name='input_char_ids_b')

	elif FLAGS.model_type in ["match_pyramid"]:
		receiver_tensors = {
			"input_ids_a":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_a'),
			"input_ids_b":tf.placeholder(tf.int32, [None, FLAGS.max_length], name='input_ids_b'),
			"label_ids":tf.placeholder(tf.int32, [None], name='label_ids')
		}

		if FLAGS.with_char == "char":
			receiver_tensors["input_char_ids_a"] = tf.placeholder(tf.int32, [None, FLAGS.char_limit, FLAGS.max_length], name='input_char_ids_a')
			receiver_tensors["input_char_ids_b"] = tf.placeholder(tf.int32, [None, FLAGS.char_limit, FLAGS.max_length], name='input_char_ids_b')

	return receiver_tensors
