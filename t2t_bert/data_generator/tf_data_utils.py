import tensorflow as tf
import numpy as np
import copy
import collections

"""
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')
features['tensor_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=tensors[i].shape))
tf_features = tf.train.Features(feature= features)
tf_example = tf.train.Example(features = tf_features)
tf_serialized = tf_example.SerializeToString()
writer.write(tf_serialized)
"""

def _decode_record(record, name_to_features):
	"""Decodes a record to a TensorFlow example.

	name_to_features = {
				"input_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"masked_lm_positions":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
						tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
						tf.FixedLenFeature([1], tf.int64),
		}

	"""
	example = tf.parse_single_example(record, name_to_features)

	# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
	# So cast all int64 to int32.
	for name in list(example.keys()):
		t = example[name]
		if t.dtype == tf.int64:
			t = tf.to_int32(t)
		example[name] = t

	return example

def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

def train_input_fn(input_file, _parse_fn, name_to_features,
		params):
	dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
	dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
	dataset = dataset.shuffle(buffer_size=params.get("buffer_size", 1024),
							seed=np.random.randint(0,1e10,1)[0],
							reshuffle_each_iteration=True)
	dataset = dataset.batch(params.get("batch_size", 32))
	dataset = dataset.repeat(params.get("epoch", 100))
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()
	return features

def eval_input_fn(input_file, _parse_fn, name_to_features,
		params):
	dataset = tf.data.TFRecordDataset(input_file, buffer_size=params.get("buffer_size", 100))
	dataset = dataset.map(lambda x:_parse_fn(x, name_to_features))
	dataset = dataset.batch(params.get("batch_size", 32))
	dataset = dataset.repeat(1)
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()
	return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def _truncate_seq_pair_v1(tokens_a, tokens_b, max_length, rng):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break

		trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
		assert len(trunc_tokens) >= 1

		# We want to sometimes truncate from the front and sometimes from the
		# back to add more randomness and avoid biases.
		if rng.random() < 0.5:
		 	del trunc_tokens[0]
		else:
			trunc_tokens.pop()

def _truncate_seq(tokens_a, max_length, rng):
	while True:
		total_length = len(tokens_a)
		trunc_tokens = tokens_a
		if total_length <= max_length:
			break
		if rng.random() < 0.5:
			del trunc_tokens[0]
		else:
			trunc_tokens.pop()

def input_fn_builder(input_files,
										 max_seq_length,
										 max_predictions_per_seq,
										 is_training,
										 num_cpu_threads=4):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]

		name_to_features = {
				"input_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"masked_lm_positions":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
						tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
						tf.FixedLenFeature([1], tf.int64),
		}

		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		if is_training:
			d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
			d = d.repeat()
			d = d.shuffle(buffer_size=len(input_files))

			# `cycle_length` is the number of parallel files that get read.
			cycle_length = min(num_cpu_threads, len(input_files))

			# `sloppy` mode means that the interleaving is not exact. This adds
			# even more randomness to the training pipeline.
			d = d.apply(
					tf.contrib.data.parallel_interleave(
							tf.data.TFRecordDataset,
							sloppy=is_training,
							cycle_length=cycle_length))
			d = d.shuffle(buffer_size=100)
		else:
			d = tf.data.TFRecordDataset(input_files)
			# Since we evaluate for a fixed number of steps we don't want to encounter
			# out-of-range exceptions.
			d = d.repeat()

		# We must `drop_remainder` on training because the TPU requires fixed
		# size dimensions. For eval, we assume we are evaluating on the CPU or GPU
		# and we *don't* want to drop the remainder, otherwise we wont cover
		# every sample.
		d = d.apply(
				tf.contrib.data.map_and_batch(
						lambda record: _decode_record(record, name_to_features),
						batch_size=batch_size,
						num_parallel_batches=num_cpu_threads,
						drop_remainder=True))
		return d

	return input_fn

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
												 orig_answer_text):
	"""Returns tokenized answer spans that better match the annotated answer."""

	# The SQuAD annotations are character based. We first project them to
	# whitespace-tokenized words. But then after WordPiece tokenization, we can
	# often find a "better match". For example:
	#
	#   Question: What year was John Smith born?
	#   Context: The leader was John Smith (1895-1943).
	#   Answer: 1895
	#
	# The original whitespace-tokenized answer will be "(1895-1943).". However
	# after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
	# the exact answer, 1895.
	#
	# However, this is not always possible. Consider the following:
	#
	#   Question: What country is the top exporter of electornics?
	#   Context: The Japanese electronics industry is the lagest in the world.
	#   Answer: Japan
	#
	# In this case, the annotator chose "Japan" as a character sub-span of
	# the word "Japanese". Since our WordPiece tokenizer does not split
	# "Japanese", we just use "Japanese" as the annotation. This is fairly rare
	# in SQuAD, but does happen.
	tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

	for new_start in range(input_start, input_end + 1):
		for new_end in range(input_end, new_start - 1, -1):
			text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
			if text_span == tok_answer_text:
				return (new_start, new_end)

	return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
	"""Check if this is the 'max context' doc span for the token."""

	# Because of the sliding window approach taken to scoring documents, a single
	# token can appear in multiple documents. E.g.
	#  Doc: the man went to the store and bought a gallon of milk
	#  Span A: the man went to the
	#  Span B: to the store and bought
	#  Span C: and bought a gallon of
	#  ...
	#
	# Now the word 'bought' will have two scores from spans B and C. We only
	# want to consider the score with "maximum context", which we define as
	# the *minimum* of its left and right context (the *sum* of left and
	# right context will always be the same, of course).
	#
	# In the example the maximum context for 'bought' would be span C since
	# it has 1 left context and 3 right context, while span B has 4 left context
	# and 0 right context.
	best_score = None
	best_span_index = None
	for (span_index, doc_span) in enumerate(doc_spans):
		end = doc_span.start + doc_span.length - 1
		if position < doc_span.start:
			continue
		if position > end:
			continue
		num_left_context = position - doc_span.start
		num_right_context = end - position
		score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
		if best_score is None or score > best_score:
			best_score = score
			best_span_index = span_index

	return cur_span_index == best_span_index

def create_masked_lm_predictions_piece(tokens, masked_lm_prob,
							max_predictions_per_seq, vocab_words, rng):
	"""Creates the predictions for the masked LM objective."""

	cand_indexes = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]":
			continue
		cand_indexes.append(i)

	rng.shuffle(cand_indexes)

	output_tokens = list(tokens)

	masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # pylint: disable=invalid-name

	num_to_predict = min(max_predictions_per_seq,
											 max(1, int(round(len(tokens) * masked_lm_prob))))

	masked_lms = []
	covered_indexes = set()
	for index in cand_indexes:
		if len(masked_lms) >= num_to_predict:
			break
		if index in covered_indexes:
			continue
		covered_indexes.add(index)

		masked_token = None
		# 80% of the time, replace with [MASK]
		if rng.random() < 0.8:
			masked_token = "[MASK]"
		else:
			# 10% of the time, keep original
			if rng.random() < 0.5:
				masked_token = tokens[index]
			# 10% of the time, replace with random word
			else:
				masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

		output_tokens[index] = masked_token

		masked_lms.append(masked_lm(index=index, label=tokens[index]))

	masked_lms = sorted(masked_lms, key=lambda x: x.index)

	masked_lm_positions = []
	masked_lm_labels = []
	for p in masked_lms:
		masked_lm_positions.append(p.index)
		masked_lm_labels.append(p.label)

	return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_masked_lm_predictions_wwm(tokens, masked_lm_prob,
						max_predictions_per_seq, vocab_words, rng):
	"""Creates the predictions for the masked LM objective."""

	cand_indexes = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]":
			continue
		# Whole Word Masking means that if we mask all of the wordpieces
		# corresponding to an original word. When a word has been split into
		# WordPieces, the first token does not have any marker and any subsequence
		# tokens are prefixed with ##. So whenever we see the ## token, we
		# append it to the previous set of word indexes.
		#
		# Note that Whole Word Masking does *not* change the training code
		# at all -- we still predict each WordPiece independently, softmaxed
		# over the entire vocabulary.
		if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
				token.startswith("##")):
			cand_indexes[-1].append(i)
		else:
			cand_indexes.append([i])

	rng.shuffle(cand_indexes)

	output_tokens = list(tokens)

	num_to_predict = min(max_predictions_per_seq,
											 max(1, int(round(len(tokens) * masked_lm_prob))))

	masked_lms = []
	covered_indexes = set()
	for index_set in cand_indexes:
		if len(masked_lms) >= num_to_predict:
			break
		# If adding a whole-word mask would exceed the maximum number of
		# predictions, then just skip this candidate.
		if len(masked_lms) + len(index_set) > num_to_predict:
			continue
		is_any_index_covered = False
		for index in index_set:
			if index in covered_indexes:
				is_any_index_covered = True
				break
		if is_any_index_covered:
			continue
		for index in index_set:
			covered_indexes.add(index)

			masked_token = None
			# 80% of the time, replace with [MASK]
			if rng.random() < 0.8:
				masked_token = "[MASK]"
			else:
				# 10% of the time, keep original
				if rng.random() < 0.5:
					masked_token = tokens[index]
				# 10% of the time, replace with random word
				else:
					masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

			output_tokens[index] = masked_token

			masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
	assert len(masked_lms) <= num_to_predict
	masked_lms = sorted(masked_lms, key=lambda x: x.index)

	masked_lm_positions = []
	masked_lm_labels = []
	for p in masked_lms:
		masked_lm_positions.append(p.index)
		masked_lm_labels.append(p.label)

	return (output_tokens, masked_lm_positions, masked_lm_labels)
