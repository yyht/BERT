from model.match_pyramid import match_pyramid
import tensorflow as tf

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

	match_matrix = model._semantic_interaction(input_ids_a, 
								input_char_ids_a, 
								input_ids_b, 
								input_char_ids_b,
								emb_seq_a, 
								enc_seq_a, 
								emb_seq_b, 
								enc_seq_b,
								is_training,
								reuse=reuse)

	print("==match_matrix shape==", match_matrix.get_shape())

	model._semantic_aggerate(match_matrix, 
							is_training,
							dpool_index=features.get("dpool_index", None),
							reuse=reuse)

	return model