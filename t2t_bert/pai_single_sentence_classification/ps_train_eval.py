# -*- coding: utf-8 -*-
import sys,os
sys.path.append("..")

from train_eval_fn import train_eval_fn

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string(
	"opt_type", "ps",
	"Input TF example files (can be a glob or comma separated).")

def monitored_sess(worker_count, 
				task_index, 
				cluster, 
				is_chief, 
				target,
				init_checkpoint,
				train_file,
				dev_file,
				checkpoint_dir):
	
	sess_config = tf.ConfigProto()

	if worker_count >= 1 and FLAGS.opt_type == "ps":
		available_worker_device = "/job:worker/task:%d" % (task_index)
		with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
			train_eval_fn(worker_count, 
							task_index, 
							is_chief, 
							target,
							init_checkpoint,
							train_file,
							dev_file,
							checkpoint_dir)
	else:
		train_eval_fn(worker_count, 
					task_index, 
					is_chief, 
					target,
					init_checkpoint,
					train_file,
					dev_file,
					checkpoint_dir)

if __name__ == "__main__":
	monitored_sess(worker_count=1,
					task_index=0,
					is_chief=0,
					target="",
					init_checkpoint=FLAGS.init_checkpoint,
					train_file=FLAGS.train_file,
					dev_file=FLAGS.dev_file,
					checkpoint_dir=FLAGS.model_output)
