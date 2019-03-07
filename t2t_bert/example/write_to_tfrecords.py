import tensorflow as tf
from data_generator import tf_data_utils
from example.feature_writer import ClassifierFeatureWriter, SpanFeatureWriter, MultiChoiceFeatureWriter
from data_generator import data_feature_classifier
from data_generator import data_distillation_feature_classifier
from data_generator import data_feature_mrc
from data_generator import tokenization
import collections
from data_generator import pair_data_feature_classifier
from example.feature_writer import PairClassifierFeatureWriter
from example.feature_writer import PairPreTrainingFeature
from example.feature_writer import DistillationEncoderFeatureWriter

def convert_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	feature_writer = ClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = None
		if example.text_b:
			try:
				tokens_b = tokenizer.tokenize(example.text_b)
			except:
				print("==token b error==", example.text_b, ex_index)
				break

		if tokens_b:
			tf_data_utils._truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)

		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]
		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)

		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		try:

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
		except:
			print(len(input_ids), max_seq_length, ex_index, "length error")
			break

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1
		if ex_index < 5:
			print(tokens)
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			tf.logging.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			tf.logging.info("label: {} (id = {})".format(example.label, label_id))

		feature = data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()

def convert_span_mrc_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	"""Loads a data file into a list of `InputBatch`s."""

	unique_id = 1000000000
	feature_writer = SpanFeatureWriter(output_file, is_traiing=False)
	for (example_index, example) in enumerate(examples):
		query_tokens = tokenizer.tokenize(example.question_text)

		if len(query_tokens) > max_query_length:
			query_tokens = query_tokens[0:max_query_length]

		tok_to_orig_index = []
		orig_to_tok_index = []
		all_doc_tokens = []
		for (i, token) in enumerate(example.doc_tokens):
			orig_to_tok_index.append(len(all_doc_tokens))
			sub_tokens = tokenizer.tokenize(token)
			for sub_token in sub_tokens:
				tok_to_orig_index.append(i)
				all_doc_tokens.append(sub_token)

		tok_start_position = None
		tok_end_position = None
		if is_training:
			tok_start_position = orig_to_tok_index[example.start_position]
			if example.end_position < len(example.doc_tokens) - 1:
				tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
			else:
				tok_end_position = len(all_doc_tokens) - 1
			(tok_start_position, tok_end_position) = tf_data_utils._improve_answer_span(
					all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
					example.orig_answer_text)

		# The -3 accounts for [CLS], [SEP] and [SEP]
		max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

		# We can have documents that are longer than the maximum sequence length.
		# To deal with this we do a sliding window approach, where we take chunks
		# of the up to our max length with a stride of `doc_stride`.
		_DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
				"DocSpan", ["start", "length"])
		doc_spans = []
		start_offset = 0
		while start_offset < len(all_doc_tokens):
			length = len(all_doc_tokens) - start_offset
			if length > max_tokens_for_doc:
				length = max_tokens_for_doc
			doc_spans.append(_DocSpan(start=start_offset, length=length))
			if start_offset + length == len(all_doc_tokens):
				break
			start_offset += min(length, doc_stride)

		for (doc_span_index, doc_span) in enumerate(doc_spans):
			tokens = []
			token_to_orig_map = {}
			token_is_max_context = {}
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in query_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for i in range(doc_span.length):
				split_token_index = doc_span.start + i
				token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

				is_max_context = tf_data_utils._check_is_max_context(doc_spans, doc_span_index,
																							 split_token_index)
				token_is_max_context[len(tokens)] = is_max_context
				tokens.append(all_doc_tokens[split_token_index])
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			start_position = None
			end_position = None
			if is_training:
				# For training, if our document chunk does not contain an annotation
				# we throw it out, since there is nothing to predict.
				doc_start = doc_span.start
				doc_end = doc_span.start + doc_span.length - 1
				if (example.start_position < doc_start or
						example.end_position < doc_start or
						example.start_position > doc_end or example.end_position > doc_end):
					continue

				doc_offset = len(query_tokens) + 2
				start_position = tok_start_position - doc_start + doc_offset
				end_position = tok_end_position - doc_start + doc_offset

			if example_index < 20:
				tf.logging.info("*** Example ***")
				tf.logging.info("unique_id: %s" % (unique_id))
				tf.logging.info("example_index: %s" % (example_index))
				tf.logging.info("doc_span_index: %s" % (doc_span_index))
				tf.logging.info("tokens: %s" % " ".join(
						[tokenization.printable_text(x) for x in tokens]))
				tf.logging.info("token_to_orig_map: %s" % " ".join(
						["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
				tf.logging.info("token_is_max_context: %s" % " ".join([
						"%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
				]))
				tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				tf.logging.info(
						"input_mask: %s" % " ".join([str(x) for x in input_mask]))
				tf.logging.info(
						"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
				if is_training:
					answer_text = " ".join(tokens[start_position:(end_position + 1)])
					tf.logging.info("start_position: %d" % (start_position))
					tf.logging.info("end_position: %d" % (end_position))
					tf.logging.info(
							"answer: %s" % (tokenization.printable_text(answer_text)))

			feature = data_feature_mrc.InputFeatures(
					unique_id=unique_id,
					example_index=example_index,
					doc_span_index=doc_span_index,
					tokens=tokens,
					token_to_orig_map=token_to_orig_map,
					token_is_max_context=token_is_max_context,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					start_position=start_position,
					end_position=end_position)

			feature_writer.process_feature(feature)
			unique_id += 1
	feature_writer.close()

def convert_multichoice_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	feature_writer = MultiChoiceFeatureWriter(output_file, is_training=False)
	for (ex_index, example) in enumerate(examples):
		question_text = tokenizer.tokenize(example.question_text)
		context_text = tokenizer.tokenize(example.doc_tokens)

		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		question_context = question_text + context_text
		choice_token_ids = []
		choice_segment_ids = []
		choice_mask = []
		choice_tokens = []
		for answer in example.answer_choice:
			answer_text = tokenizer.tokenize(answer)
			tf_data_utils._truncate_seq_pair(question_context, answer_text, max_seq_length-3)

			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)

			for token in question_context:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for token in answer_text:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length

			choice_token_ids.extend(input_ids)
			choice_segment_ids.extend(segment_ids)
			choice_mask.extend(input_mask)
			choice_tokens.extend(tokens)

		assert len(choice_token_ids) == max_seq_length * len(example.answer_choice)

		if ex_index < 5:
			tf.logging.info("*** Example ***")
			tf.logging.info("tokens: {}".format(choice_token_ids))
			tf.logging.info("choice: {} answer {}".format(example.choice, example.answer_choice))

			# tf.logging.info("*** Example ***")
			# tf.logging.info("qas_id: %s" % (example.qas_id))
			# tf.logging.info("tokens: %s" % " ".join(
			# 		[tokenization.printable_text(x) for x in tokens]))
			# tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			# tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			# tf.logging.info(
			# 		"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			# tf.logging.info("choice: {} answer {}".format(example.choice, example.answer_choice))
		
		feature = data_feature_mrc.InputFeatures(
					unique_id=example.qas_id,
					input_ids=choice_token_ids,
					input_mask=choice_mask,
					segment_ids=choice_segment_ids,
					choice=example.choice)
		feature_writer.process_feature(feature)
	feature_writer.close()

		
def convert_interaction_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	feature_writer = PairClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = tokenizer.tokenize(example.text_b)
			
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

		if len(tokens_b) > max_seq_length - 2:
			tokens_b = tokens_b[0:(max_seq_length - 2)]

		def get_input(input_tokens):
			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)

			for token in input_tokens:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			return [tokens, input_ids, 
					input_mask, segment_ids]

		[tokens_a,
		input_ids_a, 
		input_mask_a, 
		segment_ids_a] = get_input(tokens_a)

		[tokens_b,
		input_ids_b, 
		input_mask_b, 
		segment_ids_b] = get_input(tokens_b)

		try:
			assert len(input_ids_a) == max_seq_length
			assert len(input_mask_a) == max_seq_length
			assert len(segment_ids_a) == max_seq_length

			assert len(input_ids_b) == max_seq_length
			assert len(input_mask_b) == max_seq_length
			assert len(segment_ids_b) == max_seq_length

		except:
			print(len(input_ids_a), input_ids_a, max_seq_length, ex_index, "length error")
			break

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1
		if ex_index < 5:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens_a: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_a]))
			tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
			tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
			tf.logging.info(
					"segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))

			tf.logging.info("tokens_b: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_b]))
			tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
			tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
			tf.logging.info(
					"segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))

			tf.logging.info("label: {} (id = {})".format(example.label, label_id))
		
		feature = pair_data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids_a=input_ids_a,
					input_mask_a=input_mask_a,
					segment_ids_a=segment_ids_a,
					input_ids_b=input_ids_b,
					input_mask_b=input_mask_b,
					segment_ids_b=segment_ids_b,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()

def convert_interaction_classifier_examples_to_features_v1(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	feature_writer = PairClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = tokenizer.tokenize(example.text_b)
			
		if len(tokens_a) > max_seq_length:
			tokens_a = tokens_a[0:(max_seq_length)]
		else:
			tokens_a += ["[PAD]"]*(max_seq_length-len(tokens_a))

		if len(tokens_b) > max_seq_length:
			tokens_b = tokens_b[0:(max_seq_length)]
		else:
			tokens_b += ["[PAD]"]*(max_seq_length-len(tokens_b))

		def get_input(tokens_a, tokens_b):
			tokens = []
			segment_ids = []
			input_mask = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			input_mask.append(1)
			for token in tokens_a:
				tokens.append(token)
				segment_ids.append(0)
				if token == "[PAD]":
					input_mask.append(0)
				else:
					input_mask.append(1)
			tokens.append("[SEP]")
			segment_ids.append(0)
			input_mask.append(1)
			for token in tokens_b:
				if token == "[PAD]":
					input_mask.append(0)
				else:
					input_mask.append(1)
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)
			input_mask.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# Zero-pad up to the sequence length.
			while len(input_ids) < 2*max_seq_length+3:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			return [tokens, input_ids, 
					input_mask, segment_ids]

		[tokens_ab,
		input_ids_ab, 
		input_mask_ab, 
		segment_ids_ab] = get_input(tokens_a, tokens_b)

		[tokens_ba,
		input_ids_ba, 
		input_mask_ba, 
		segment_ids_ba] = get_input(tokens_b, tokens_a)

		try:
			assert len(input_ids_ab) == 2*max_seq_length+3
			assert len(input_mask_ab) == 2*max_seq_length+3
			assert len(segment_ids_ab) == 2*max_seq_length+3

			assert len(input_ids_ba) == 2*max_seq_length+3
			assert len(input_mask_ba) == 2*max_seq_length+3
			assert len(segment_ids_ba) == 2*max_seq_length+3

		except:
			print(len(input_ids_ab), input_ids_ab, 2*max_seq_length+3, ex_index, "length error")
			break

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1
		if ex_index < 5:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens_ab: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_ab]))
			tf.logging.info("input_ids_ab: %s" % " ".join([str(x) for x in input_ids_ab]))
			tf.logging.info("input_mask_ab: %s" % " ".join([str(x) for x in input_mask_ab]))
			tf.logging.info(
					"segment_ids_ab: %s" % " ".join([str(x) for x in segment_ids_ab]))

			tf.logging.info("tokens_ba: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_ba]))
			tf.logging.info("input_ids_ba: %s" % " ".join([str(x) for x in input_ids_ba]))
			tf.logging.info("input_mask_ba: %s" % " ".join([str(x) for x in input_mask_ba]))
			tf.logging.info(
					"segment_ids_ba: %s" % " ".join([str(x) for x in segment_ids_ba]))

			tf.logging.info("label: {} (id = {})".format(example.label, label_id))
		
		feature = pair_data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids_a=input_ids_ab,
					input_mask_a=input_mask_ab,
					segment_ids_a=segment_ids_ab,
					input_ids_b=input_ids_ba,
					input_mask_b=input_mask_ba,
					segment_ids_b=segment_ids_ba,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()

def convert_pair_order_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file):

	feature_writer = PairClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = tokenizer.tokenize(example.text_b)

		tf_data_utils._truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)

		def get_input(input_tokens_a, input_tokens_b):
			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)

			for token in input_tokens_a:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			for token in input_tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_seq_length:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)

			return [tokens, input_ids, 
					input_mask, segment_ids]

		[tokens_a_,
		input_ids_a, 
		input_mask_a, 
		segment_ids_a] = get_input(tokens_a, tokens_b)

		[tokens_b_,
		input_ids_b, 
		input_mask_b, 
		segment_ids_b] = get_input(tokens_b, tokens_a)

		try:
			assert len(input_ids_a) == max_seq_length
			assert len(input_mask_a) == max_seq_length
			assert len(segment_ids_a) == max_seq_length

			assert len(input_ids_b) == max_seq_length
			assert len(input_mask_b) == max_seq_length
			assert len(segment_ids_b) == max_seq_length

		except:
			print(len(input_ids_a), input_ids_a, max_seq_length, ex_index, "length error")
			break

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1
		if ex_index < 5:
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens_a: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_a_]))
			tf.logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
			tf.logging.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
			tf.logging.info(
					"segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))

			tf.logging.info("tokens_b: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens_b_]))
			tf.logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
			tf.logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
			tf.logging.info(
					"segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))

			tf.logging.info("label: {} (id = {})".format(example.label, label_id))
		
		feature = pair_data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids_a=input_ids_a,
					input_mask_a=input_mask_a,
					segment_ids_a=segment_ids_a,
					input_ids_b=input_ids_b,
					input_mask_b=input_mask_b,
					segment_ids_b=segment_ids_b,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()

def convert_classifier_examples_with_rule_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, rule_detector, 
											output_file):

	feature_writer = ClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = None
		if example.text_b:
			try:
				tokens_b = tokenizer.tokenize(example.text_b)
			except:
				print("==token b error==", example.text_b, ex_index)
				break

		if tokens_b:
			tf_data_utils._truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)

		else:
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]
		tokens = []
		segment_ids = [0]
		rule_ids = rule_detector.infer(tokens_a) # input is tokenized list
		tokens.append("[CLS]")

		for index, token in enumerate(tokens_a):
			tokens.append(token)
			segment_ids.append(rule_ids[index])
		tokens.append("[SEP]")
		segment_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		try:

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
		except:
			print(len(input_ids), max_seq_length, ex_index, "length error")
			break

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1
		if ex_index < 5:
			print(tokens)
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			tf.logging.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			tf.logging.info("label: {} (id = {})".format(example.label, label_id))

		feature = data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()

def convert_distillation_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file, with_char,
											char_len):

	feature_writer = DistillationEncoderFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_b = None
		if example.text_b:
			try:
				tokens_b = tokenizer.tokenize(example.text_b)
			except:
				print("==token b error==", example.text_b, ex_index)
				break

		if len(tokens_a) > max_seq_length:
			tokens_a = tokens_a[0:(max_seq_length)]
		if tokens_b:
			if len(tokens_b) > max_seq_length:
				tokens_b = tokens_b[0:(max_seq_length)]
			input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b, max_seq_length)
			if with_char == "char":
				input_char_ids_b = tokenizer.covert_tokens_to_char_ids(tokens_b, 
											max_seq_length, 
											char_len=char_len)
		else:
			input_ids_b = None
			input_char_ids_b = None

		input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a, max_seq_length)
		if with_char == "char":
			input_char_ids_a = tokenizer.covert_tokens_to_char_ids(tokens_a, 
											max_seq_length, 
											char_len=char_len)
		else:
			input_char_ids_a = None		

		if len(example.label) == 1:
			label_id = label_dict[example.label[0]]
		else:
			label_id = [0] * len(label_dict)
			for item in example.label:
				label_id[label_dict[item]] = 1

		try:
			label_probs = example.label_probs
		except:
			label_probs = [0.0]*len(label_dict)

		try:
			label_ratio = example.label_ratio
		except:
			label_ratio = 0.0

		if ex_index < 5:
			print(tokens_a)
			tf.logging.info("*** Example ***")
			tf.logging.info("guid: %s" % (example.guid))
			tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids_a]))
			if input_char_ids_a:
				tf.logging.info("input_ids: %s" % " ".join([str(x) for token in input_char_ids_a for x in token ]))
			if input_ids_b:
				tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids_b]))
			if input_char_ids_b:
				tf.logging.info("input_ids: %s" % " ".join([str(x) for token in input_char_ids_b for x in token ]))
			tf.logging.info("label probs {}".format(label_probs))
			tf.logging.info("label_ratio {}".format(label_ratio))
			tf.logging.info("label: {} (id = {})".format(example.label, label_id))
		
		feature = data_distillation_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids_a=input_ids_a,
					input_ids_b=input_ids_b,
					input_char_ids_a=input_char_ids_a,
					input_char_ids_b=input_char_ids_b,
					label_ids=label_id,
					label_probs=label_probs,
					label_ratio=label_ratio)
		feature_writer.process_feature(feature)
	feature_writer.close()
