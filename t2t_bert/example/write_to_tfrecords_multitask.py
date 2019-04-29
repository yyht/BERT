import tensorflow as tf
from data_generator import tf_data_utils
from data_generator import data_feature_classifier

from data_generator import tokenization
import collections

from example.feature_writer import MultitaskFeatureWriter

def convert_multitask_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file,
											task_type,
											task_type_dict):

	feature_writer = MultitaskFeatureWriter(output_file, is_training=False)

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
		feature_writer.process_feature(feature, task_type, task_type_dict)
	feature_writer.close()

