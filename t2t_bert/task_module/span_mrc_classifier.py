import tensorflow as tf
from utils.bert import bert_utils
from loss import loss_utils
from utils.bert import albert_modules

def span_extraction_classifier(config, 
							sequence_output,
							start_positions, 
							end_positions, 
							input_span_mask):

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

	# apply output mask
	adder           = (1.0 - tf.cast(input_span_mask, tf.float32)) * -10000.0
	start_logits   += adder
	end_logits     += adder

	def compute_loss(logits, positions):
		on_hot_pos    = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
		log_probs     = tf.nn.log_softmax(logits, axis=-1)
		loss          = -tf.reduce_mean(tf.reduce_sum(on_hot_pos * log_probs, axis=-1))
		return loss

	start_positions = features["start_positions"]
	end_positions   = features["end_positions"]

	start_loss  = compute_loss(start_logits, start_positions)
	end_loss    = compute_loss(end_logits, end_positions)
	total_loss  = (start_loss + end_loss) / 2

	return (total_loss, start_logits, end_logits)

