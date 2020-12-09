
import tensorflow as tf
from utils.bert import bert_utils
from utils.bert import bert_modules
from model.match_pyramid import mp_cnn

def get_article_sentence(input_tensor, 
					sentence_positions,
					sentence_positions_mask,
					**kargs):
	input_shape_list = bert_utils.get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape_list[0]
	seq_length = input_shape_list[1]
	hidden_dims = input_shape_list[2]

	# [batch_size, sentence_num, seq_length]
	sentence_positions_mapping = tf.cast(tf.one_hot(sentence_positions, seq_length), 
											dtype=tf.float32)

	# [batch_size, sentence_num, hidden_dims]
	input_tensor = tf.einsum("aid,aki->akd", input_tensor, sentence_positions_mapping)
	input_tensor *= tf.cast(sentence_positions_mask[:, None], dtype=tf.float32)
	return input_tensor

def article_sentence_alignment(config, input_tensor_a, 
								sentence_positions_a, 
								sentence_positions_mask_a,
								input_tensor_b, 
								sentence_positions_b, 
								sentence_positions_mask_b,
								**kargs):
	
	input_a_feat_set = get_article_sentence(input_tensor_a, 
					sentence_positions_a,
					sentence_positions_mask_a,
					**kargs)

	input_b_feat_set = get_article_sentence(input_tensor_b, 
					sentence_positions_b,
					sentence_positions_mask_b,
					**kargs)

	input_a_feat_set_norm = tf.nn.l2_normalize(input_a_feat_set, axis=-1)
	input_b_feat_set_norm = tf.nn.l2_normalize(input_b_feat_set, axis=-1)

	similarity_matrix = tf.einsum("abc,adc->abd", input_a_feat_set_norm, input_b_feat_set_norm)
	aggerate_feature = mp_cnn._mp_semantic_feature_layer(config,
													similarity_matrix, 
													kargs.get("dpool_index", None),
													reuse=tf.AUTO_REUSE)
	return aggerate_feature