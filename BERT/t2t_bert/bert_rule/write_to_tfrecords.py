import tensorflow as tf
from data_generator import tf_data_utils
from example.feature_writer import ClassifierFeatureWriter, SpanFeatureWriter, MultiChoiceFeatureWriter
from data_generator import data_feature_classifier
from data_generator import data_feature_mrc
from data_generator import tokenization
import collections

def convert_classifier_examples_to_features(examples, label_dict, 
											max_seq_length,
											tokenizer, output_file, 
											rule_matcher, background_label):

	feature_writer = ClassifierFeatureWriter(output_file, is_training=False)

	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

		tokens_a_rule = rule_matcher.parse(tokens_a, background_label)

		tokens = []
		tokens.append("[CLS]")
		rule_ids.append(label_dict[background_label])
		rule_ids = [label_dict[rule[0]] for rule in tokens_a_rule]

		for token in tokens_a:
			tokens.append(token)

		tokens.append("[SEP]")
		rule_ids.append(label_dict[background_label])

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			rule_ids.append(label_dict[background_label])

		try:

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(rule_ids) == max_seq_length
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
					"rule_ids: %s" % " ".join([str(x) for x in rule_ids]))
			tf.logging.info("label: {} (id = {})".format(example.label, label_id))

		feature = data_feature_classifier.InputFeatures(
					guid=example.guid,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=rule_ids,
					label_ids=label_id)
		feature_writer.process_feature(feature)
	feature_writer.close()