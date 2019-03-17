from model.match_pyramid import match_pyramid

def match_pyramid_encoder(model_config, features, labels, 
			mode, target, reuse=None):

	if mode == tf.estimator.ModeKeys.TRAIN:
		is_training = tf.constant(True)
	else:
		is_training = tf.constant(False)

	input_ids_a = features["input_ids_a"]
	input_char_ids_a = features.get("input_char_ids_a", None)

	input_ids_b = features["input_ids_b"]
	input_char_ids_b = features.get("input_char_ids_b", None)

	model = match_pyramid.MatchPyramid(model_config)
	[emb_seq_a, enc_seq_a, 
	emb_seq_b, enc_seq_b] = model._semantic_encode(input_ids_a, 
											input_char_ids_a, 
											input_ids_b, 
											input_char_ids_b,
											is_training,
											reuse=reuse)

	model._semantic_interaction(input_ids_a, 
								input_char_ids_a, 
								input_ids_b, 
								input_char_ids_b,
								emb_seq_a, 
								enc_seq_a, 
								emb_seq_b, 
								enc_seq_b,
								is_training,
								reuse=reuse)

	model._semantic_aggerate(match_matrix, is_training, reuse=reuse, 
								dpool_index=features.get("dpool_index", None))

	return model