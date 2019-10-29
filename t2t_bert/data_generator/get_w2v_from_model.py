import tensorflow as tf
import os

flags = tf.flags

FLAGS = flags.FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string("buckets", "", "oss buckets")

flags.DEFINE_string(
	"model_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_gensim_w2v_path", None,
	"Input TF example files (can be a glob or comma separated).")

def main(_):
	model_path = os.path.join(FLAGS.buckets, FLAGS.model_path)
	vocab_path = os.path.join(FLAGS.buckets, FLAGS.vocab_path)
	output_gensim_w2v_path = os.path.join(FLAGS.buckets, FLAGS.output_gensim_w2v_path)

	init_vars = tf.train.list_variables(model_path)
	for name, shape in init_vars:
		if 'word_embeddings' in name:
			array = tf.train.load_variable(model_path, name)
			
	with open(vocab_path) as frobj:
		vocab = []
		for line in frobj:
			vocab.append(line.strip())

	with open(output_gensim_w2v_path, "w") as fwobj:
		fwobj.write(" ".join([len(vocab), array.shape[1]])+"\n")
		for term, vec in zip(vocab, array):
			vec = vec.tolist()
			fwobj.write(" ".join([term]+vec)+"\n")