import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils

def classifier(config, pooled_output, 
						num_labels, labels,
						dropout_prob,
						ratio_weight=None,
						**kargs):

	output_layer = pooled_output

	hidden_size = output_layer.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

	logits = tf.matmul(output_layer, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	logits = tf.nn.log_softmax(logits)

	if config.get("label_type", "single_label") == "single_label":
		if config.get("loss", "entropy") == "entropy":
			print("==standard cross entropy==")
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))
		elif config.get("loss", "entropy") == "focal_loss":
			print("==multi_label focal loss==")
			per_example_loss, _ = loss_utils.focal_loss_multi_v1(config,
														logits=logits, 
														labels=labels)
			
		try:
			per_example_loss = loss_utils.weighted_loss_ratio(
											config, per_example_loss, 
											labels, ratio_weight)
			loss = tf.reduce_sum(per_example_loss)
			print(" == applying weighted loss == ")
		except:
			loss = tf.reduce_mean(per_example_loss)

		if config.get("with_center_loss", "no") == "center_loss":
			print("==apply with center loss==")
			center_loss, _ = loss_utils.center_loss_v2(config,
											features=pooled_output, 
											labels=labels)
			loss += center_loss * config.get("center_loss_coef", 1e-3)

		return (loss, per_example_loss, logits)
	elif config.get("label_type", "single_label") == "multi_label":
		logits = tf.log_sigmoid(logits)
		per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))
		per_example_loss = tf.reduce_sum(per_example_loss, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, per_example_loss, logits)
	else:
		raise NotImplementedError()

def multi_choice_classifier(config, pooled_output, 
		num_labels, labels, dropout_prob):
	output_layer = pooled_output
	
	final_hidden_shape = bert_utils.get_shape_list(output_layer, 
								expected_rank=2)

	print(final_hidden_shape, "====multi-choice shape====")

	output_layer = tf.reshape(output_layer, 
								[-1,
								num_labels,
								final_hidden_shape[-1]]) # batch x num_choices x hidden_dim

	hidden_size = output_layer.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)
	logits = tf.einsum("abc,c->ab", output_layer, output_weights)
	logits = tf.nn.bias_add(logits, output_bias) # batch x num_labels

	if config.get("loss_type", "entropy") == "focal_loss":
		per_example_loss = loss_utils.focal_loss_multi_v1(logits=logits, 
													labels=labels)
	else:
		per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))
	loss = tf.reduce_mean(per_example_loss)

	return (loss, per_example_loss, logits)

def span_extraction_classifier(config, sequence_output,
		start_positions, end_positions):

	final_hidden_shape = bert_modules.get_shape_list(sequence_output, 
								expected_rank=3)

	batch_size = final_hidden_shape[0]
	seq_length = final_hidden_shape[1]
	hidden_size = final_hidden_shape[2]

	output_weights = tf.get_variable(
			"cls/mrc_span/output_weights", [2, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"cls/mrc_span/output_bias", [2], initializer=tf.zeros_initializer())

	final_hidden_matrix = tf.reshape(final_hidden,
																	 [batch_size * seq_length, hidden_size])
	logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	logits = tf.reshape(logits, [batch_size, seq_length, 2])
	logits = tf.transpose(logits, [2, 0, 1])

	unstacked_logits = tf.unstack(logits, axis=0)

	(start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

	def compute_loss(logits, positions):

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(positions))
		loss = tf.reduce_mean(loss)
		return loss
		
	start_loss = compute_loss(start_logits, start_positions)
	end_loss = compute_loss(end_logits, end_positions)

	loss = start_loss + end_loss

	return (loss, start_logits, end_logits)

def seq_label_classifier(config, sequence_output, 
										num_labels, labels,
										label_weights):
	final_hidden_shape = bert_modules.get_shape_list(sequence_output, 
								expected_rank=3)

	batch_size = final_hidden_shape[0]
	seq_length = final_hidden_shape[1]
	hidden_size = final_hidden_shape[2]

	output_weights = tf.get_variable(
			"cls/seq_label/output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"cls/seq_label/output_bias", [num_labels], initializer=tf.zeros_initializer())

	final_hidden_matrix = tf.reshape(final_hidden,
																	 [batch_size * seq_length, hidden_size])

	logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias) # batch x seq , num_labels

	# batch x seq , num_labels
	logits = tf.reshape(logits, [batch_size, seq_length, num_labels])

	per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits,
												labels=tf.stop_gradient(labels))
	losses *= label_weights
	loss = tf.reduce_mean(losses)

	return (loss, logits, per_example_loss)

def interaction_classifier(config, output_lst, 
						num_labels, labels,
						dropout_prob,
						ratio_weight=None):
	
	assert len(output_lst) == 2

	seq_output_a = output_lst[0]
	seq_output_b = output_lst[1]

	# batch x (hidden x 4)
	repres = tf.concat([seq_output_a, seq_output_b,
						seq_output_a - seq_output_b,
						seq_output_a * seq_output_b],
						axis=-1)

	hidden_size = repres.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(repres, keep_prob=1 - dropout_prob)

	logits = tf.matmul(output_layer, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	if config.get("label_type", "single_label") == "single_label":
		if config.get("loss", "entropy") == "entropy":
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))

		elif config.get("loss", "entropy") == "focal_loss":
			per_example_loss = loss_utils.focal_loss_multi_v1(config,
														logits=logits, 
														labels=labels)
		try:
			per_example_loss = loss_utils.weighted_loss_ratio(
											config, per_example_loss, 
											labels, ratio_weight)
			loss = tf.reduce_sum(per_example_loss)
		except:
			loss = tf.reduce_mean(per_example_loss)
		
		return (loss, per_example_loss, logits)

def order_classifier(config, output_lst, 
						num_labels, labels,
						dropout_prob,
						ratio_weight=None):
	
	assert len(output_lst) == 2

	seq_output_a = output_lst[0]
	seq_output_b = output_lst[1]

	# batch x (hidden x 2)
	# repres = tf.concat([seq_output_a, seq_output_b],
	# 					axis=-1)

	repres = seq_output_a + seq_output_b

	hidden_size = repres.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(repres, keep_prob=1 - dropout_prob)

	logits = tf.matmul(output_layer, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	if config.get("label_type", "single_label") == "single_label":
		if config.get("loss", "entropy") == "entropy":
			print("==apply entropy loss==")
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))

		elif config.get("loss", "entropy") == "focal_loss":
			tf.logging.info("===apply multi-class focal loss===")
			print("===apply multi-class focal loss===")
			per_example_loss = loss_utils.focal_loss_multi_v1(config,
														logits=logits, 
														labels=labels)
		try:
			per_example_loss = loss_utils.weighted_loss_ratio(
											config, per_example_loss, 
											labels, ratio_weight)
			loss = tf.reduce_sum(per_example_loss)
		except:
			loss = tf.reduce_mean(per_example_loss)
		
		return (loss, per_example_loss, logits)

def order_classifier_v1(config, output_lst, 
						num_labels, labels,
						dropout_prob,
						ratio_weight=None):
	
	assert len(output_lst) == 2

	seq_output_a = output_lst[0]
	seq_output_b = output_lst[1]

	# batch x (hidden x 2)
	# repres = tf.concat([seq_output_a, seq_output_b],
	# 					axis=-1)

	repres = seq_output_a + seq_output_b

	hidden_size = repres.shape[-1].value

	repres = tf.layers.dense(repres, hidden_size, 
							activation=tf.nn.tanh,
							name="output_dense")

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(repres, keep_prob=1 - dropout_prob)

	logits = tf.matmul(output_layer, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	if config.get("label_type", "single_label") == "single_label":
		if config.get("loss", "entropy") == "entropy":
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))

		elif config.get("loss", "entropy") == "focal_loss":
			tf.logging.info("===apply multi-class focal loss===")
			print("===apply multi-class focal loss===")
			per_example_loss = loss_utils.focal_loss_multi_v1(config,
														logits=logits, 
														labels=labels)
		try:
			per_example_loss = loss_utils.weighted_loss_ratio(
											config, per_example_loss, 
											labels, ratio_weight)
			loss = tf.reduce_sum(per_example_loss)
		except:
			loss = tf.reduce_mean(per_example_loss)
		
		return (loss, per_example_loss, logits)

def distributed_classifier(config, pooled_output, 
						num_labels, labels,
						dropout_prob,
						ratio_weight=None):

	output_layer = pooled_output

	hidden_size = output_layer.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

	logits = tf.matmul(output_layer, output_weights, transpose_b=True)
	logits = tf.nn.bias_add(logits, output_bias)

	if config.get("label_type", "single_label") == "single_label":
		if config.get("loss", "entropy") == "entropy":
			per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))
		elif config.get("loss", "entropy") == "focal_loss":
			per_example_loss = loss_utils.focal_loss_multi_v1(config,
														logits=logits, 
														labels=labels)
		loss = tf.reduce_mean(per_example_loss)

		return (loss, per_example_loss, logits)
	elif config.get("label_type", "single_label") == "multi_label":
		logits = tf.log_sigmoid(logits)
		per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
												logits=logits, 
												labels=tf.stop_gradient(labels))
		per_example_loss = tf.reduce_mean(per_example_loss, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
		return (loss, per_example_loss, logits)
	else:
		raise NotImplementedError()

def siamese_classifier(config, pooled_output, num_labels,
						labels, dropout_prob,
						ratio_weight=None):

	if config.get("output_layer", "interaction") == "interaction":
		print("==apply interaction layer==")
		repres_a = pooled_output[0]
		repres_b = pooled_output[1]

		output_layer = tf.concat([repres_a, repres_b, tf.abs(repres_a-repres_b), repres_a*repres_b], axis=-1)
		hidden_size = output_layer.shape[-1].value

		output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

		output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

		output_layer = tf.nn.dropout(output_layer, keep_prob=1 - dropout_prob)

		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)

		print("==logits shape==", logits.get_shape())

		if config.get("label_type", "single_label") == "single_label":
			if config.get("loss", "entropy") == "entropy":
				per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
													logits=logits, 
													labels=tf.stop_gradient(labels))
			elif config.get("loss", "entropy") == "focal_loss":
				per_example_loss, _ = loss_utils.focal_loss_multi_v1(config,
															logits=logits, 
															labels=labels)
			print("==per_example_loss shape==", per_example_loss.get_shape())
			loss = tf.reduce_mean(per_example_loss)

			return (loss, per_example_loss, logits)
		elif config.get("label_type", "single_label") == "multi_label":
			logits = tf.log_sigmoid(logits)
			per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
													logits=logits, 
													labels=tf.stop_gradient(labels))
			per_example_loss = tf.reduce_mean(per_example_loss, axis=-1)
			loss = tf.reduce_mean(per_example_loss)
			return (loss, per_example_loss, logits)
		else:
			raise NotImplementedError()



