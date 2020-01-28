# -*- coding: utf-8 -*-

import requests
import json
import tensorflow as tf
import sys,os

father_path = os.path.join(os.getcwd())
print(father_path, "==father path==")

def find_bert(father_path):
	if father_path.split("/")[-1] == "BERT":
		return father_path

	output_path = ""
	for fi in os.listdir(father_path):
		if fi == "BERT":
			output_path = os.path.join(father_path, fi)
			break
		else:
			if os.path.isdir(os.path.join(father_path, fi)):
				find_bert(os.path.join(father_path, fi))
			else:
				continue
	return output_path

bert_path = find_bert(father_path)
t2t_bert_path = os.path.join(bert_path, "t2t_bert")
sys.path.extend([bert_path, t2t_bert_path])

print(sys.path)

from distributed_single_sentence_classification import tf_serving_data_prepare as single_sent_data_prepare
from distributed_pair_sentence_classification import tf_serving_data_prepare as pair_sent_data_prepare

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"buckets", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"vocab", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_bool(
	"do_lower_case", True,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"url", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"port", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"model_name", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"signature_name", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"keyword_path", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"background_label", "正常",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"label_dict", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"segment_id_type", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"add_word_path", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"delete_word_path", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"input_data", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"output_path", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"task_type", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"model_type", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"tokenizer", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"with_char", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"versions", "",
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_integer(
	"max_seq_length", 64,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

def main(_):

	if FLAGS.task_type == "pair_sentence_classification":
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab)
		corpus_path = os.path.join(FLAGS.buckets, FLAGS.input_data)
		print(corpus_path, vocab_path)
		feed_dict = pair_sent_data_prepare.get_feeddict(FLAGS, vocab_path, corpus_path)
		output_path = os.path.join(FLAGS.buckets, FLAGS.output_path)
	elif FLAGS.task_type == "single_sentence_classification":
		vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab)
		corpus_path = os.path.join(FLAGS.buckets, FLAGS.input_data)
		print(corpus_path, vocab_path)
		feed_dict = single_sent_data_prepare.get_feeddict(FLAGS, vocab_path, corpus_path)
		output_path = os.path.join(FLAGS.buckets, FLAGS.output_path)

	results = requests.post("http://%s:%s/v1/models/%s/versions/%s:predict" % (FLAGS.url, 
															FLAGS.port, FLAGS.model_name, 
															FLAGS.versions), 
															json=feed_dict)
	try:
		with tf.gfile.Open(output_path, "w") as fwobj:
			pred_lst = results.content.decode()
			json.dump(pred_lst, fwobj)
		print(results.content.decode())
	except:
		predict_info = []
		print(results.content.decode())

if __name__ == "__main__":
	tf.app.run()
