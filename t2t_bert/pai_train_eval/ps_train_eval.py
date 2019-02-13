# -*- coding: utf-8 -*-
import sys,os
sys.path.append("..")

from pai_single_sentence_classification import ps_train_eval

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string('worker_hosts', '', 'must be list')
flags.DEFINE_string('job_name', '', 'must be in ("", "worker", "ps")')
flags.DEFINE_integer('task_index', 0, '')
flags.DEFINE_string("ps_hosts", "", "must be list")
flags.DEFINE_string("buckets", "", "oss buckets")

def main(_):
	ps_spec = FLAGS.ps_hosts.split(",")
	worker_spec = FLAGS.worker_hosts.split(",")
	cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
	worker_count=len(worker_spec)

	is_chief = FLAGS.task_index == 0

	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
							protocol="grpc++")

	init_checkpoint = os.path.join(FLAGS.buckets, FLAGS.init_checkpoint)
	train_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	dev_file = os.path.join(FLAGS.buckets, FLAGS.train_file)
	checkpoint_dir = os.path.join(FLAGS.buckets, FLAGS.model_output)
	
	# join the ps server
	if FLAGS.job_name == "ps":
		server.join()
	try:
		ps_train_eval.monitored_sess(worker_count=worker_count, 
			task_index=FLAGS.task_index, 
			cluster=cluster, 
			is_chief=is_chief, 
			target=server.target,
			init_checkpoint=init_checkpoint,
			train_file=train_file,
			dev_file=dev_file,
			checkpoint_dir=checkpoint_dir)

	except Exception, e:
		print("catch a exception: %s" % e.message)

if __name__ == "__main__":
	tf.app.run()