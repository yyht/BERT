import tensorflow as tf
from utils.bert import bert_utils

def _to_term_frequency(x, vocab_size):
	"""Creates a SparseTensor of term frequency for every doc/term pair.
	Args:
		x : a SparseTensor of int64 representing string indices in vocab.
		vocab_size: A scalar int64 Tensor - the count of vocab used to turn the
				string into int64s including any OOV buckets.
	Returns:
		a SparseTensor with the count of times a term appears in a document at
				indices <doc_index_in_batch>, <term_index_in_vocab>,
				with size (num_docs_in_batch, vocab_size).
	"""
	# Construct intermediary sparse tensor with indices
	# [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
	vocab_size = tf.convert_to_tensor(value=vocab_size, dtype=tf.int64)
	split_indices = tf.cast(
			tf.split(x.indices, axis=1, num_or_size_splits=2), dtype=tf.int64)
	expanded_values = tf.cast(tf.expand_dims(x.values, 1), dtype=tf.int64)
	next_index = tf.concat(
			[split_indices[0], split_indices[1], expanded_values], axis=1)

	next_values = tf.ones_like(x.values)
	expanded_vocab_size = tf.expand_dims(vocab_size, 0)
	next_shape = tf.concat(
			[x.dense_shape, expanded_vocab_size], 0)

	next_tensor = tf.SparseTensor(
			indices=tf.cast(next_index, dtype=tf.int64),
			values=next_values,
			dense_shape=next_shape)

	# Take the intermediary tensor and reduce over the term_index_in_doc
	# dimension. This produces a tensor with indices [<doc_id>, <term_id>]
	# and values [count_of_term_in_doc] and shape batch x vocab_size
	term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)

	dense_doc_sizes = tf.cast(
			tf.sparse.reduce_sum(
					tf.SparseTensor(
							indices=x.indices,
							values=tf.ones_like(x.values),
							dense_shape=x.dense_shape), 1),
			dtype=tf.float64)

	gather_indices = term_count_per_doc.indices[:, 0]
	gathered_doc_sizes = tf.gather(dense_doc_sizes, gather_indices)

	term_frequency = (
			tf.cast(term_count_per_doc.values, dtype=tf.float64) /
			tf.cast(gathered_doc_sizes, dtype=tf.float64))
	term_count = tf.cast(term_count_per_doc.values, dtype=tf.float64)

	sparse_term_freq = tf.SparseTensor(
							indices=term_count_per_doc.indices,
							values=term_frequency,
							dense_shape=term_count_per_doc.dense_shape)

	sparse_term_count = tf.SparseTensor(
							indices=term_count_per_doc.indices,
							values=term_count,
							dense_shape=term_count_per_doc.dense_shape)

	return sparse_term_freq, sparse_term_count

def _to_sparse(x):
	tensor_shape = bert_utils.get_shape_list(x, expected_rank=[2])
	idx = tf.where(tf.not_equal(x, 0))
	# Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
	sparse = tf.SparseTensor(idx, tf.gather_nd(x, idx), tensor_shape)
	return sparse

def _to_vocab_range(x, vocab_size):
	"""Enforces that the vocab_ids in x are positive."""
	output = tf.SparseTensor(
			indices=x.indices,
			values=tf.mod(x.values, vocab_size),
			dense_shape=x.dense_shape)
	return output

def sparse_idf2dense(sparse_term_freq, sparse_term_count):
	dense_term_freq = tf.sparse.to_dense(sparse_term_freq)
	dense_term_count = tf.sparse.to_dense(sparse_term_count)
	return dense_term_freq, dense_term_count
