import tensorflow as tf
import os
import horovod.tensorflow as hvd

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## Required parameters

flags.DEFINE_string(
	"buckets", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"meta", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string(
	"input_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_checkpoint", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"log_directory", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_meta_graph", None,
	"Input TF example files (can be a glob or comma separated).")

def main(_):

	graph = tf.Graph()
	with graph.as_default():
		import json

		meta = os.path.join(FLAGS.buckets, FLAGS.meta)
		input_checkpoint = os.path.join(FLAGS.buckets, FLAGS.input_checkpoint)
		output_checkpoint = os.path.join(FLAGS.buckets, FLAGS.output_checkpoint)
		output_meta_graph = os.path.join(FLAGS.buckets, FLAGS.output_meta_graph)

		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=True))
				
		saver = tf.train.import_meta_graph(meta, clear_devices=True)
		# saver = tf.train.import_meta_graph(FLAGS.meta)
		print("==succeeded in loading meta graph==")
		saver.restore(sess, input_checkpoint)
		print("==succeeded in loading model==")
		saver.save(sess, output_checkpoint)
		print("==succeeded in restoring model==")
		saver.export_meta_graph(output_meta_graph, clear_devices=True)

if __name__ == "__main__":
	tf.app.run()


