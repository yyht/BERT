import tensorflow as tf

def naive_embedding_dropout(config, embedding_matrix):
	embedding_matrix = tf.nn.dropout(embedding_matrix, 
								keep_prob=1-config.get('embedding_dropout', 0.1), 
								noise_shape=[config.vocab_size,1])
	return embedding_matrix

def effcient_embedding_dropout(config, embedding_matrix, input_ids):
	batch_size = tf.shape(input_ids)[0]
	uniq_ids, indices = tf.unique(tf.reshape(input_ids, [-1]))
	rand_mask = tf.random_uniform([batch_size, tf.size(uniq_ids)], dtype=embedding_matrix.dtype)

	mask_indices = tf.stack([batch_wise, uniq_ids_wise], axis=-1)
	binary_mask = tf.floor(tf.gather_nd(rand_mask, mask_indices) + keep_prob)

	# apply mask and scale
	dropped_embeddings = embed_ids * tf.expand_dims(binary_mask, axis=-1) / keep_prob

	