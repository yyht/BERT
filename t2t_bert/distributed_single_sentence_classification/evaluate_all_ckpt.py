
import tensorflow as tf
import os

def evalue_all_ckpt(checkpoint_dir, task_name, estimator_fn):

	###### Actual eval loop ######
	num_mins_waited = 0
	eval_results = []

	# Gather a list of checkpoints to evaluate
	steps_and_files = []
	try:
		filenames = tf.gfile.ListDirectory(checkpoint_dir)
	except tf.errors.NotFoundError:
		filenames = []
		tf.logging.info("`checkpoint_dir` does not exist yet...")

	for filename in filenames:
		if filename.endswith(".index"):
			ckpt_name = filename[:-6]
			cur_filename = os.path.join(checkpoint_dir, ckpt_name)
			global_step = int(cur_filename.split("-")[-1])
			# if (global_step <= last_eval_step or
			# 		global_step > FLAGS.eval_end_step):
			# 	continue
			tf.logging.info("[{}] Add {} to eval list.".format(global_step,
																												 cur_filename))
			steps_and_files.append([global_step, cur_filename])

	# # Get empty list of checkpoints
	# if not steps_and_files:
	# 	# Training job is done: stop evaluation
	# 	if tf.io.gfile.exists(os.path.join(eval_model_dir, "done")):
	# 		break
	# 	# Wait for 60 seconds
	# 	else:
	# 		time.sleep(60)
	# 		num_mins_waited += 1.0
	# 		tf.logging.info("Waited {:.1f} mins".format(num_mins_waited))
	# else:
	# 	num_mins_waited = 0

	# Evaluate / Predict / Submit the current list of checkpoints
	for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
		##### Validation
		ret = estimator_fn(filename)
		# ret = estimator.evaluate(
		# 		input_fn=eval_input_fn,
		# 		steps=eval_steps,
		# 		checkpoint_path=filename)

		ret["step"] = global_step
		ret["path"] = filename

		# if task_name in ["cola", "mrpc", "qqp"]:
		# 	tp, fp, tn, fn = (ret["eval_tp"], ret["eval_fp"], ret["eval_tn"],
		# 										ret["eval_fn"])
		# 	ret["eval_f1"] = _compute_metric_based_on_keys(
		# 			key="f1", tp=tp, fp=fp, tn=tn, fn=fn)
		# 	ret["eval_corr"] = _compute_metric_based_on_keys(
		# 			key="corr", tp=tp, fp=fp, tn=tn, fn=fn)

		eval_results.append(ret)

		# Log current result
		tf.logging.info("=" * 80)
		log_str = "Eval step {} | ".format(global_step)
		for key, val in eval_results[-1].items():
			log_str += "{} {} | ".format(key, val)
		tf.logging.info(log_str)
		tf.logging.info("=" * 80)

	##### Log the best validation result
	key_func = lambda x: x["acc"]
	# if task_name == "sts-b":
	# 	key_func = lambda x: x["eval_pearsonr"]
	# if task_name == "cola":
	# 	key_func = lambda x: x["eval_corr"]
	# if task_name in ["mrpc", "qqp"]:
	# 	key_func = lambda x: x["eval_f1"] + x["eval_accuracy"]
	eval_results.sort(key=key_func, reverse=True)
	tf.logging.info("=" * 80)
	log_str = "Best eval result | "
	for key, val in eval_results[0].items():
		log_str += "{} {} | ".format(key, val)
	tf.logging.info(log_str)